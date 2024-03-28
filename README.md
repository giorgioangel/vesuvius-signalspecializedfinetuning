This repository contains a notebook that implements Signal Specialized parametrization for the Vesuvius Challenge, as described in https://hhoppe.com/ssp.pdf . The parametrization is implemented as a fine-tuning over an already existing parametrization (flattening).
First, we import an existing mesh .obj, then we multiply the UV map by the proper dimension taken from the .png or .tif of the segment (sorry, I was not doing it automatically).
Since Signal-Specialized parametrization requires the values of the voxels in the scroll volume, we first create a dictionary containing the required voxels and then we extract their values from the scroll volume.
Eventually, the signal-specialized objective is minimized using the AdamW optimizer and torch.

The second part of the notebook performs the following steps: 1) importing a mesh, 2) performing a intrinsic delaunay triangulation with igl (applying this algorithm https://link.springer.com/article/10.1007/s00607-007-0249-8 ) 3) flattening with SLIM ( https://github.com/giorgioangel/slim-flatboi ) ( https://igl.ethz.ch/projects/slim/SLIM2017.pdf ), 4) signal-specialized fine-tuning.
