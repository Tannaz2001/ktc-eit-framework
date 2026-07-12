# Code for KTC2023 EIT challenge (Plug&Play + mask)


## Brief description of the algorithm
This repository contains the Python files required by the organisers of the EIT Kuopio challenge 2023 which implements a Plug&Play approach with a post-processing step where synthetic (data-independent) masks are superimposed on the obtained reconstruction to improve results.

Training was performed using GT simulated data with random polygons of different shapes, size and postive/negative contrast. In more detail, we created synthetic shapes of:
- rectangles with sides of different length and with different orientation within the domain
- squares with side of different length and with different orientation within the domain
- generic triangles with sides of different lenghts
- generic quadrilaterals with sides of different lengthts
- circles of different radii and centres
- horseshoe shapes of different sides and orientations
- star-like shapes with 3-4-5 tips with different centres and orientations
For rectangles, squares, circles and star-like shapes we further allowed the possibility of having holes of different size.
All shapes were assigned to a random negative/positive value within fixed ranges to simulate resistive/conductive inclusions. We allowed a number of inclusions shaped as above equal to 1, 2, 3, making sure that all shapes do not intersect and are note placed too close to the boundary.

Here are some simulated inclusions (different shapes, size, number, position):
![Fig2](https://github.com/lucala00/KTC2023_E2E/assets/49308207/7143a902-d650-4c2e-bfe6-da21a19a9550)
![fig5](https://github.com/lucala00/KTC2023_E2E/assets/49308207/ef9111ad-8e03-46dc-83ca-2f548cebebb3)
![Figure_3](https://github.com/lucala00/KTC2023_E2E/assets/49308207/1960ec95-f80b-4b35-b3a7-6d8e7ed69e1c)
![fig1](https://github.com/lucala00/KTC2023_E2E/assets/49308207/c034634f-363c-4c60-99fc-a8aa7ae59a43)

Associated noisy versions were computed running a proximal Gauss-Newton method (on meshes) with a pre-trained graph neural denoiser in place of the standard proximal map for few (5) iterations on the corresponding measurements simulated using the forward model provided.

To improved the segmentation performed on the reconstructions obtained, we incorporated a post-processing step where an artificial mask setting to 0 the reconstructed values in a thin portion of the boundary associated to the arc of electrodes removed, where artefacts appeared. Examples associated to the masks used for difficulty level 3 and 5 are reported here:

![mask_difficulty3](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/385bde9a-76b8-410e-8e40-c4a6c4b0721c)
![mask_difficulty5](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/a25ae97a-456c-46e3-a777-44a02e03f5db)

We considered a graph-U-net denoiser extending the CNN-based denoiser to non-Euclidian manifold domain. It relies on the Graph U-net architecture, a U-Net-like architecture for graph data which allows high-level feature encoding and decoding for network embedding. It is based on a convolution graph kernel and gPool and gUnpool layers. The pool (gPool) operation samples some nodes to form a smaller graph based on their scalar projection values on a trainable projection vector. As an inverse operation of gPool, the unpooling (gUnpool) operation restores the graph to its original structure with the help of locations of nodes selected in the corresponding gPool layer.

The GU-Net-denoiser as well as the Graph-U-Net can be formalized as a composition of layers, where each layer is characterized by the composition of a graph convolution, which is nothing but a ReLU activation function σ and a gPool/gUnpool operator.


## Authors:
- Tatiana Bubba, University of Bath, UK, tab73'at'bath.ac.uk
- Luca Calatroni, CNRS, FR, calatroni'at'i3s.unice.fr
- Damiana Lazzaro, University of Bologna, IT, damiana.lazzaro'at'unibo.it 
- Serena Morigi, University of Bologna, IT, serena.morigi'at'unibo.it 
- Luca Ratti, University of Bologna, IT, luca.ratti5'at'unibo.it
- Matteo Santacesaria, University of Genoa, IT, matteo.Santacesaria'at'unige.it 
- Julian Tachella, CNRS, FR, julian.tachella'at'ens-lyon.fr

## Installation instructions and requirements

Installation enviroment_pnp for pytorch 2.1 and cuda 12.1

```conda create -n  environment_pnp```

```conda activate environment_pnp```

```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```

```pip install torch_geometric==2.3.1```

```pip install torch_scatter torch_sparse torch_cluster  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html  (To be replaced with your version (example: ${TORCH}=2.1.0 and ${CUDA}=cu121)```

```conda install matplotlib```

```pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv```

We created a script main.py to reconstruct the inclusions provided for training from voltage measurements:

```python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level```

For the denoiser, the learned weights are stored in the file
``` weights_denoiser.pth```

## Examples

For the TrainingData provided by the challenge organisers

![truth1](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/3e58fdcf-3300-4e46-9584-89c8e6cec608)
![truth2](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/f8332093-fc0e-4647-ab6b-f2acc2f9dc61)
![truth3](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/06455860-ea2a-4d01-a9a2-9fbee1c89133)
![truth4](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/d5111d85-fe37-42b9-8cb8-d1bc8f03a576)

After applying the masks described above to the computed reconstruction the results obtaind are reported below, together with the corresponding score:

![Examples_PnP_synthetic](https://github.com/lucala00/KTC2023_PNPmasked/assets/49308207/2a00966f-9300-4e98-b944-d156bef70d22)


## Reference work: 
* Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Deep-plug-and-play proximal Gauss-Newton method with applications to nonlinear, ill-posed inverse problems. Inverse Problems and Imaging, 2023, 17(6): 1226-1248. doi: 10.3934/ipi.2023014
* Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Learning Nonlinear Electrical Impedance Tomography. J Sci Comput, 2022, 90(58). https://doi.org/10.1007/s10915-021-01716-4
