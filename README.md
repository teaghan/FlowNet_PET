<p align="left">
  <img width="550" height="167" src="./docs/XCITElogo.png">
</p>

# FlowNet-PET
Unsupervised Learning to Perform Respiratory Motion Correction in PET Imaging

## Dependencies

-[PyTorch](http://pytorch.org/): `pip install torch torchvision`
  - We used version 1.10.0, but I would assume that other versions will still work.

-h5py: `pip install h5py`

-configparser: `pip install configparser`

## Data download

The data can be downloaded [here](https://zenodo.org/record/5851646) or possibly by just clicking this [link](https://zenodo.org/record/5851646/files/validation_data.zip?download=1).

Once downloaded, unzip the file and place each of the subdirectories within the [patient directory](./patients/). For instance, after doing this you should have a directory tree `PET_MonteCarlo/patients/nema_simu`.

## Training the Network


### Analysis notebooks
