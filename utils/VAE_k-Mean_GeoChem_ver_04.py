# !pip install spectral
# !pip install rasterio
# Basic import
import matplotlib.pyplot as plt
import numpy as np
from time import time
import rasterio as rio
from sklearn.preprocessing import minmax_scale
from sklearn import cluster
from sklearn.decomposition import PCA


import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# Importing the data
data_raster = rio.open('drive/MyDrive/VAE_GeoChem/Delamerian_ASTER.tif')
data_test   = rio.open('drive/MyDrive/VAE_GeoChem/Wilcannia_RockUnits_30m.tif')
# print(data_raster.meta)

## Visualizing the data
# Reading and enhancing
data_array = data_raster.read() # reading the data
vmin, vmax = np.nanpercentile(data_array, (5,95)) # 5-95% pixel values stretch

data_array_test = data_test.read() # reading the data
vmin, vmax = np.nanpercentile(data_array_test, (5,95)) # 5-95% pixel values stretch
# Plotting the enhanced image
# fig = plt.figure(figsize=[20,20])
# plt.axis('off')
# plt.imshow(data_array[0, :, :], vmin=vmin, vmax=vmax)
# plt.show()


## Reshaping the Train data from brc to rcb
# Creating an empty array with the same dimension and data type
imgxyb = np.empty((data_raster.height, data_raster.width, data_raster.count), data_raster.meta['dtype'])
# Looping through the bands to fill the empty array
for band in range(imgxyb.shape[2]):
    imgxyb[:,:,band] = data_raster.read(band+1)

# Reshaping the input data from rcb to samples and features
data_reshaped = imgxyb.reshape(imgxyb.shape[0]*imgxyb.shape[1], -1)
# Scaling
data_reshaped = minmax_scale(data_reshaped, feature_range=(0, 1), axis=0, copy=False)
data_reshaped.shape

## Reshaping the Test data from brc to rcb
# Creating an empty array with the same dimension and data type
imgxyb_test = np.empty((data_test.height, data_test.width, data_test.count), data_test.meta['dtype'])
# Looping through the bands to fill the empty array
for band in range(imgxyb_test.shape[2]):
    imgxyb_test[:,:,band] = data_test.read(band+1)

# Reshaping the input data from rcb to samples and features
data_reshaped_test = imgxyb_test.reshape(imgxyb_test.shape[0]*imgxyb_test.shape[1], -1)
# Scaling
data_reshaped_test = minmax_scale(data_reshaped, feature_range=(0, 1), axis=0, copy=False)
data_reshaped_test.shape


# divide data in Train - Validation - Test
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, data_reshaped_test, test_size=0.3, random_state=42)
X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize Data
sc = StandardScaler()
X_tr_std = sc.fit_transform(X_tr)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)



# Let’s set up this AE:

encoder = keras.models.Sequential([
    keras.layers.Dense(8, input_shape=[9]),
    keras.layers.Dense(7, input_shape=[8]),
    keras.layers.Dense(6, input_shape=[7]),
    keras.layers.Dense(5, input_shape=[6]),
])

decoder = keras.models.Sequential([
    keras.layers.Dense(6, input_shape=[5]),
    keras.layers.Dense(7, input_shape=[6]),
    keras.layers.Dense(8, input_shape=[7]),
    keras.layers.Dense(9, input_shape=[8]),
])

autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss='mse', optimizer = keras.optimizers.SGD(learning_rate=0.01))

history = autoencoder.fit(X_tr_std,X_tr_std, epochs=20,validation_data=(X_valid_std,X_valid_std),
                         callbacks=[keras.callbacks.EarlyStopping(patience=10)])

codings = encoder.predict(X_tr_std)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.savefig("summarize history for loss.jpg")
