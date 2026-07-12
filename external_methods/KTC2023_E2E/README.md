# Code for KTC2023 EIT challenge (end-to-end)


## Brief description of the method
This repository contains the Python files required by the organisers of the EIT Kuopio challenge 2023 which implements an end-to-end approach.

Training was performed using GT simulated data with random polygons of different shapes, size and postive/negative contrast. In more detail, we created synthetic shapes of:
- rectangles with sides of different length and with different orientation within the domain
- squares with side of different length and with different orientation within the domain
- generic triangles with sides of different lenghts
- generic quadrilaterals with sides of different lengthts
- circles of different radii and centres
- horseshoe shapes of different sides and orientations
- star-like shapes with 3-4-5 tips with different centres and orientations
For rectangles, squares, circles and star-like shapes we further allowed the possibility of having holes of different size.
All shapes were assigned to a random negative/positive value within fixed ranges to simulate resistive/conductive inclusions. We allowed a number of inclusions shaped as above equal to 1, 2, 3, making sure that all shapes do not intersect and are not placed too close to the boundary.

Here are some simulated inclusions (different shapes, size, number, position):
![Fig2](https://github.com/lucala00/KTC2023_E2E/assets/49308207/7143a902-d650-4c2e-bfe6-da21a19a9550)
![fig5](https://github.com/lucala00/KTC2023_E2E/assets/49308207/ef9111ad-8e03-46dc-83ca-2f548cebebb3)
![Figure_3](https://github.com/lucala00/KTC2023_E2E/assets/49308207/1960ec95-f80b-4b35-b3a7-6d8e7ed69e1c)
![fig1](https://github.com/lucala00/KTC2023_E2E/assets/49308207/c034634f-363c-4c60-99fc-a8aa7ae59a43)

For each inclusion, the corresponding GT labeling was considered by assigning to class 0 the background elements, to class 1 the resistive inclusions and to class 2 the conductive inclusions.
Using the forward EIT model provided, for each element of the training data a corresponding measurement vector of difficulty from 1 to 7 is created.

The end-to-end procedure takes as input those measurement vectors and maps them directly into a segmented image with associated labels 0, 1, 2. It is based on a convolutional UNet architecture with 4 scales, where the first (learnable) linear layer maps the measurements to image space. This first linear layer is initialized using a Tikhonov reconstruction with regularization parameter 0.001.  For all measurement vectors, the network is trained so as to match minimise the cross entropy loss between the labeling of the reconstructed image with the GT labeled data defined above.

## Authors:
- Tatiana Bubba, University of Bath, UK, tab73 AT bath.ac.uk
- Luca Calatroni, CNRS, FR, calatroni AT i3s.unice.fr
- Damiana Lazzaro, University of Bologna, IT, damiana.lazzaro AT unibo.it 
- Serena Morigi, University of Bologna, IT, serena.morigi AT unibo.it 
- Luca Ratti, University of Bologna, IT, luca.ratti5 AT unibo.it
- Matteo Santacesaria, University of Genoa, IT, matteo.santacesaria AT unige.it 
- Julian Tachella, CNRS, FR, julian.tachella AT ens-lyon.fr

## Installation instructions and requirements

Please use the following command to create the virtual environment with the required packages:

```conda env create -f environment_pnp.yml```

We created a script main.py to reconstruct the inclusions provided for training from voltage measurements:

```python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level```

The same architecture was employed for all levels of difficulties. Training was repeated to better adapt to each level of difficulty. For each level of difficulty, the learned parameters of the network are available in at the path

```/models/difficulty_{j}_learnedlinear.pth.tar ```

and uploaded directly via the call to the main file once the difficulty level is specified.

To install the latest stable release of deepinv, you can simply do:

```pip install deepinv```

You can also install the latest version of deepinv directly from github:

``` pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv``` 

## Examples

For the training data provided, we tested the approach for all level of difficulties comparing the reconstruction with the ground truth images provided. Here below the results we obtained.

![e2e](https://github.com/lucala00/KTC2023_E2E/assets/49308207/089da0f9-0e6d-45ea-9bcc-6aea5ca01884)


The ```data4``` appeared to be particularly challenging, we believe due to the low values of the measurement and the central position of the inclusion which, generally, corresponds to a more challenging problem.


## References

* Samuel Hurault, Mathieu Terris, Julian Tachella, DeepInverse: a Pytorch library for imaging with deep learning, https://deepinv.github.io.

