# AutoimplantPipe

This is a repository for the paper _"An End-to-End Pipeline for Automatic Patient-Specific Cranial Implant Design:
From CT Scans to Titanium Implants"_. The repository contains the following implementations:

1. **Automatic skull segmentation** predict 3D skulls from a given CT Scans in grayscale
2. **Autoimplant** predicting complete skull from a defective skull

## Notebooks

Notebooks folder contains notebook for automatic skull segmentation and autoimplant inferences including

- `01_autosegmentation.ipynb` is an example notebook for performing segmentation from a given Nifti file
- `02_autoimplant_prediction.ipynb` is an example for performing autoimplant inference

We provide the pretrained model on SkullBreak dataset for both segmentation and autoimplant
[here](https://github.com/biodatlab/autoimplantpipe).

## Training scripts

`scripts` folder includes the script for training models and baseline models.

- `train_skull_segmentation_pl.ipynb` is a notebook for training a segmentation model with Pytorch Lightning
- `train_autoimplant.py` is a script for training an autoimplant model

## Installation

Download the repository using `git`:

```sh
git clone https://github.com/biodatlab/autoimplantpipe
cd autoimplantpipe
```

Install dependencies and library using `pip`:

```sh
pip install -r requirements.txt  # install dependencies
pip install .  # install `autoimplant_pipeline` library
```
