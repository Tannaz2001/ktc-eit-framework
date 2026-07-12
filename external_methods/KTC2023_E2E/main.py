import deepinv as dinv
import torch
import numpy as np
import scipy as sp
from helper_fncs import Network, EIT_Dataset, get_mask
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import KTCScoring

device = dinv.utils.get_freer_gpu()
# device = torch.device('cpu')

# model
regularization = [.001]
pixels = 64

aver = False

def main(inputFolder,outputFolder,difficulty):
    Uelref = sp.io.loadmat(inputFolder + '/ref.mat')["Uelref"] #measured voltages from water chamber
    
    mask = get_mask(difficulty)
    backbone = dinv.models.UNet(in_channels=len(regularization), out_channels=3, scales=4).to(device)
    model = Network(backbone, regularization, difficulty=difficulty, device=device, pixels=pixels, train_first=True)

    p = Path(f"models/difficulty_{difficulty}_learnedlinear.pth.tar")

    model.load_state_dict(torch.load(p)['state_dict'])
    # model.load_state_dict(torch.load(p,map_location=torch.device('cpu'))['state_dict'])   # for CPU
    model.eval()


    # Get a list of .mat files in the input folder
    mat_files = glob.glob(inputFolder + '/data*.mat')
    for objectno in  range (0,len(mat_files)): #compute the reconstruction for each input file
        Uel = sp.io.loadmat(mat_files[objectno])["Uel"]
        deltaU = Uel - Uelref
        y = np.zeros((1, np.sum(mask), 1))

        y[0,:,:]=deltaU[mask,:]

        y = torch.Tensor(y).to(device)
        x_net = model(y)
        x_net = torch.argmax(x_net, dim=1, keepdim=True)
        
        # interpolate the reconstruction into a pixel image
        deltareco_pixgrid = np.reshape(x_net.detach().cpu().numpy(),[256,256])
        # fig, ax = plt.subplots()
        # cax = ax.imshow(deltareco_pixgrid, cmap="gray")
        # plt.colorbar(cax)
        # plt.axis('image')
        # plt.show()

        reconstruction = deltareco_pixgrid
        mdic = {"reconstruction": reconstruction}
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)
