import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from torchmetrics.image import TotalVariation
from PIL import Image
from utils import KTCMeshing
from utils import KTCFwd
from utils import KTCScoring
from utils import DIPAux

# Use of GPU
if torch.cuda.is_available():
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  dtype = torch.cuda.DoubleTensor
  map_location = 'cuda:0'
else:
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  dtype = torch.DoubleTensor
  map_location = 'cpu'

def solve(inputData, categoryNbr):
    img_resolution = 128
    linpoint = 0.7927
    deltaU_fac = 2
    dist_mode = 'lowpass'
    dist_sigma = 8e-3

    skip_n11=32
    num_scales=6
    skip_n33d=[16, 16, 32, 32, 64, 64]
    skip_n33u=[16, 16, 32, 32, 64, 64]

    solver_dict = get_solver_matrices(inputData, categoryNbr, linpoint)
    J = torch.from_numpy(solver_dict['J']).type(dtype)
    deltaU = torch.from_numpy(solver_dict['deltaU']).type(dtype) * deltaU_fac

    dist = calc_dist(solver_dict['coordinates'], x_d=img_resolution,
                    y_d=img_resolution, mode=dist_mode, sigma=dist_sigma)
    dist = torch.from_numpy(dist).type(dtype)

    c_mask = get_mask(img_resolution)
    c_mask = torch.from_numpy(c_mask).type(dtype)
   
    # Parameters
    input_depth = 32 #16
    input_type = 'noise'
    OPT_OVER = 'net'

    # Optimization Parameters
    OPTIMIZER = 'adam'
    pad = 'reflection'
    NET_TYPE = 'skip'
    LR = 3.5e-3
    reg_noise_std = 0.04
    num_iter=3000
    w_tv = 5e-8
    tv = TotalVariation().to(map_location)

    deblur_input = DIPAux.get_noise(input_depth,input_type,
                (img_resolution,img_resolution)).type(dtype).detach()

    deblur_net = DIPAux.get_net(input_depth, NET_TYPE, pad,
            skip_n33d = skip_n33d,
            skip_n33u = skip_n33u,
            skip_n11 = skip_n11,
            n_channels=1,
            num_scales = num_scales,
            upsample_mode='bilinear',
            need_sigmoid=False).type(dtype)

    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    p = DIPAux.get_params(OPT_OVER,deblur_net,deblur_input)

    optimizer = torch.optim.Adam(p, lr=LR)

    for i in range(num_iter):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            deblur_input = net_input_saved

        cond_pixels = c_mask * deblur_net(deblur_input)
        cond_nodes = dist @ cond_pixels[0,0].reshape((-1,1))

        total_loss = F.mse_loss(J @ cond_nodes, deltaU)
        total_loss += w_tv * tv(cond_pixels)

        total_loss.backward()

        optimizer.step()
    
    cond_pixels_np = np.flipud(cond_pixels[0,0].clone().detach().cpu().numpy())
    cond_pixels_np = np.array(Image.fromarray(cond_pixels_np).resize((256,256)))

    cond_pixels_np_segmented = segment(cond_pixels_np)

    return cond_pixels_np_segmented

def get_solver_matrices(inputData, categoryNbr, lin_point=1):
    Nel = 32  # number of electrodes
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    mat_dict = sp.io.loadmat('utils/reference_files/ref.mat') #load the reference data
    Injref = mat_dict["Injref"] #current injections
    Uelref = mat_dict["Uelref"] #measured voltages from water chamber
    Mpat = mat_dict["Mpat"] #voltage measurement pattern
    vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
    rmind = np.arange(0,2 * (categoryNbr - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl[:,ii] = 0
            vincl[jj,:] = 0

    # load premade finite element mesh (made using Gmsh, exported to Matlab and saved into a .mat file)
    mat_dict_mesh = sp.io.loadmat('utils/reference_files/Mesh_sparse.mat')
    g = mat_dict_mesh['g'] #node coordinates
    H = mat_dict_mesh['H'] #indices of nodes making up the triangular elements
    elfaces = mat_dict_mesh['elfaces'][0].tolist() #indices of nodes making up the boundary electrodes

    #Element structure
    ElementT = mat_dict_mesh['Element']['Topology'].tolist()
    for k in range(len(ElementT)):
        ElementT[k] = ElementT[k][0].flatten()
    ElementE = mat_dict_mesh['ElementE'].tolist() #marks elements which are next to boundary electrodes
    for k in range(len(ElementE)):
        if len(ElementE[k][0]) > 0:
            ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
        else:
            ElementE[k] = []

    #Node structure
    NodeC = mat_dict_mesh['Node']['Coordinate']
    NodeE = mat_dict_mesh['Node']['ElementConnection'] #marks which elements a node belongs to
    nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
    for k in range(NodeC.shape[0]):
        nodes[k].ElementConnection = NodeE[k][0].flatten()
    elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
    for k in range(len(ElementT)):
        elements[k].Electrode = ElementE[k]

    #2nd order mesh data
    H2 = mat_dict_mesh['H2']
    g2 = mat_dict_mesh['g2']
    elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
    ElementT2 = mat_dict_mesh['Element2']['Topology']
    ElementT2 = ElementT2.tolist()
    for k in range(len(ElementT2)):
        ElementT2[k] = ElementT2[k][0].flatten()
    ElementE2 = mat_dict_mesh['Element2E']
    ElementE2 = ElementE2.tolist()
    for k in range(len(ElementE2)):
        if len(ElementE2[k][0]) > 0:
            ElementE2[k] = [ElementE2[k][0][0][0], ElementE2[k][0][0][1:len(ElementE2[k][0][0])]]
        else:
            ElementE2[k] = []

    NodeC2 = mat_dict_mesh['Node2']['Coordinate']  # ok
    NodeE2 = mat_dict_mesh['Node2']['ElementConnection']  # ok
    nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
    for k in range(NodeC2.shape[0]):
        nodes2[k].ElementConnection = NodeE2[k][0].flatten()
    elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
    for k in range(len(ElementT2)):
        elements2[k].Electrode = ElementE2[k]

    Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
    Mesh2 = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)

    sigma0 = np.ones((len(Mesh.g), 1))*lin_point #linearization point
    corrlength = 1 * 0.115 #used in the prior
    var_sigma = 0.05 ** 2 #prior variance
    mean_sigma = sigma0

    # set up the forward solver for inversion
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

    vincl = vincl.T.flatten()

    # set up the noise model for inversion
    noise_std1 = 0.05;  # standard deviation for first noise component (relative to each voltage measurement)
    noise_std2 = 0.01;  # standard deviation for second noise component (relative to the largest voltage measurement)
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)

    mat_dict2 = sp.io.loadmat(inputData)
    Uel = mat_dict2["Uel"]
    deltaU = Uel[vincl] - Uelref[vincl]

    Usim = solver.SolveForward(sigma0, z) #forward solution at the linearization point
    J = solver.Jacobian(sigma0, z)

    outputs = {"J": J,
                "deltaU": deltaU,
                "coordinates":np.array([coord[0].flatten() for coord in NodeC])
              }

    return outputs

def calc_dist(nodes_coordinates, x_d=256, y_d=256, mode='nearest', sigma=5e-7):
    nodes_coordinates = nodes_coordinates.copy().T
    nodes_coordinates = nodes_coordinates[[1,0],:]
    n_elements = nodes_coordinates.shape[1]

    x_min = np.min(nodes_coordinates[0,:])
    y_min = np.min(nodes_coordinates[1,:])

    x_max = np.max(nodes_coordinates[0,:])
    y_max = np.max(nodes_coordinates[1,:])

    b = [x_min, y_min]
    a = [0, 0]
    a[0] = (x_max - x_min) / x_d
    a[1] = (y_max - y_min) / y_d

    xp = np.zeros((x_d,y_d))
    yp = np.zeros((x_d,y_d))

    for i in range(x_d):
        for j in range(y_d):
            xp[i,j] = a[0]*i + b[0]
            yp[i,j] = a[1]*j + b[1]

    xp = xp.reshape(-1)
    yp = yp.reshape(-1)

    dist = np.zeros((n_elements, y_d*x_d))

    if(mode=='nearest'):
        for i in range(n_elements):
            dist[i,:] = ((nodes_coordinates[:,i][:,None] - [xp,yp])**2).sum(axis=0)
            dist[i,:] = dist[i,:] == dist[i,:].min()

    elif(mode=='lowpass'):
        for i in range(n_elements):
            dist[i,:] = ((nodes_coordinates[:,i][:,None] - [xp,yp])**2).sum(axis=0)

        dist = np.exp(-dist**2/(2*sigma**2))
        dist /= (dist.sum(axis=1)+np.finfo(float).eps).reshape(-1,1)

    return dist

def get_mask(n_img):
    Y, X = np.mgrid[0:n_img,0:n_img]

    r = n_img//2 + 1
    xc = n_img//2 - 1
    yc = n_img//2 - 1

    c_mask = (X-xc)**2 + (Y-yc)**2 < r**2

    c_mask = c_mask[None,None,:,:]

    return c_mask

def segment(cond_pixels_np):
    level, x = KTCScoring.Otsu2(cond_pixels_np.flatten(), 256, 7)

    cond_pixels_np_segmented = np.zeros_like(cond_pixels_np)

    ind0 = cond_pixels_np < x[level[0]]
    ind1 = np.logical_and(cond_pixels_np >= x[level[0]],cond_pixels_np <= x[level[1]])
    ind2 = cond_pixels_np > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class

    match bgclass:
        case 0:
            cond_pixels_np_segmented[ind1] = 2
            cond_pixels_np_segmented[ind2] = 2
        case 1:
            cond_pixels_np_segmented[ind0] = 1
            cond_pixels_np_segmented[ind2] = 2
        case 2:
            cond_pixels_np_segmented[ind0] = 1
            cond_pixels_np_segmented[ind1] = 1

    opening_mask = sp.ndimage.binary_opening(cond_pixels_np_segmented, iterations=10)
    cond_pixels_np_segmented = opening_mask * cond_pixels_np_segmented
  
    return cond_pixels_np_segmented
