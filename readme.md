# Data-Centric Defense: Shaping Loss Landscape with Augmentations to Counter Model Inversion 

The code provided is an example of our experiments on the PPA attack. It is based on the open-source implementation of [PPA](https://github.com/LukasStruppek/Plug-and-Play-Attacks/tree/master). 

Our modifications are limited to the **dataset** component, which resides in the `datasets` folder. Specifically, we provide code that support automated surrogate selection based on attribution annotation for CelebA datasets.  For other datasets, it is feasible to employ a pretrained attribution classifier to automate the pipeline.
