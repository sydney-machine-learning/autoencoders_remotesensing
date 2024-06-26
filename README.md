[![DOI](https://zenodo.org/badge/512082635.svg)](https://zenodo.org/badge/latestdoi/512082635)

# Remote sensing stacked autoencoder and clustering framework for geological mapping


This repository provides code and supplementary materials for the paper entitled 'Remote sensing framework for lithological mapping via stacked autoencoders and clustering'. 
We present a framework based on different dimensionality reduction methods, including principal component analysis, canonical autoencoders, stacked autoencoders, and the k-means clustering algorithm to generate clustered maps using multispectral remote sensing data which are interpreted as lithological maps.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1+ high-end NVIDIA GPU for sampling and 1+ GPUs for training.
* 64-bit Python 3.9 and PyTorch 2.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Other Python libraries: `pip install click Pillow psutil requests scipy tqdm diffusers==0.26.3 accelerate==0.27.2`

## Preparing Dataset
This framework is applied to three different data types, including are in `/datasets/main_dataset/*.zip` folder or can be download from the source attached below.

A small dataset also included in the `datasets/sample_dataset.zip` folder.

1. Landsat-8 
[download here](https://www.usgs.gov/landsat-missions/landsat-data-access)

2. ASTER
[download here](https://asterweb.jpl.nasa.gov/)

4. Sentinel-2
[download here](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)


## Full dataset: 
[download here](https://www.dropbox.com/scl/fo/0k3d0pooovlsj97ztocb0/h?rlkey=colw5u0hc5tsnlt6ywolvo9i3&dl=0) 

##  Notebooks for each dataset: main run code.
* We share the code for each dataset separately including all the experiments.
* Note that the specific dataloader, data preprocessing and postprocessing should be done by users depending on particular datasets.

   `Autoencoder_Landsat8.ipynb` 
   
   `Autoencoder_ASTER.ipynb`

   `Autoencoder_Sentinel2.ipynb`


## Proposed method: flow overview.
<!--- ![image](https://github.com/sydney-machine-learning/autoencoders_remotesensing/assets/14858627/bbcd7578-679d-4c26-bd0d-39b65208ca2a)  align="right"  --->
<img src='https://github.com/sydney-machine-learning/autoencoders_remotesensing/assets/14858627/bbcd7578-679d-4c26-bd0d-39b65208ca2a' width='700'  >

## Results Elbow plot for each dataset and methods.
<!--- ![image](https://github.com/sydney-machine-learning/autoencoders_remotesensing/assets/14858627/098d427f-8872-480e-bd67-16e990181af1) --->
<img src='https://github.com/sydney-machine-learning/autoencoders_remotesensing/assets/14858627/098d427f-8872-480e-bd67-16e990181af1' width ='700'>

## License

Copyright &copy; 2024,Transitional Artificial Intelligence Research Group & AFFILIATES. All rights reserved.

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

### References


      @article{nagar2024remote,
        title={Remote sensing framework for geological mapping via stacked autoencoders and clustering},
        author={Nagar, Sandeep and Farahbakhsh, Ehsan and Awange, Joseph and Chandra, Rohitash},
        journal={arXiv preprint arXiv:2404.02180},
        year={2024}
      }


`Nagar, S., Farahbakhsh, E., Awange, J., Chandra, R., Remote sensing framework for lithological mapping via stacked autoencoders and clustering [Under Review]`
