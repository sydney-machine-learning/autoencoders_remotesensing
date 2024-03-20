import matplotlib.pyplot as plt
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromArray, imageChipsFromFile
import torch.nn as nn
import torch
import numpy as np
import random
import sklearn
# pip install pyrsgis==0.3.9
# pip install earthpy

# Defining file names
featureFile = 'Playa_Image.tif'
# labelFile = 'Playa_Image_Training.tif'

# Reading and normalizing input data
dsFeatures, arrFeatures = raster.read(featureFile, bands='all')
arrFeatures = arrFeatures.astype(float)

for i in range(arrFeatures.shape[0]):
    bandMin = arrFeatures[i][:][:].min()
    bandMax = arrFeatures[i][:][:].max()
    bandRange = bandMax-bandMin
    for j in range(arrFeatures.shape[1]):
        for k in range(arrFeatures.shape[2]):
            arrFeatures[i][j][k] = (arrFeatures[i][j][k]-bandMin)/bandRange

# Creating chips using pyrsgis
features = imageChipsFromArray(arrFeatures, x_size=7, y_size=7)

def train_test_split(features,  trainProp=0.75):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    # train_y = labels[randIndex[:sliceIndex]]
    # test_y = labels[randIndex[sliceIndex:]]
    return(train_x,  test_x, )

# Calling the function to split the data
train_x, test_x = train_test_split(features)

data_loader = torch.utils.data.DataLoader(train_x,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=0)


single_band_file = r'Playa_Image.tif' # this is a Landsat 5 TM image (7 bands stacked)
# multi_band_file = r'Playa_Image.tif' # this is a Landsat 5 TM image (7 bands stacked)

# create image chips
single_band_chips = imageChipsFromFile(single_band_file, x_size=28, y_size=28)
# multi_band_chips = imageChipsFromFile(multi_band_file, x_size=16, y_size=16)

# print(single_band_chips.shape)
# print(multi_band_chips.shape)
# print(single_band_chips[1].shape)                                          

# changin the type of data images 
single_band_chips        = np.rollaxis(single_band_chips, 3, 1)
single_band_chips_float  = single_band_chips.astype(np.float32)

single_band_chips_tensor = torch.as_tensor(single_band_chips_float)

# print(single_band_chips_tensor.dtype)

print('no of chips: bends: h: W ',single_band_chips.shape)