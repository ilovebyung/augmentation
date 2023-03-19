# check GPU
import concurrent.futures
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# Load training images
# resize and normalize data for training

image = cv2.imread('D:/data/sonics/01.jpg')
shape = image.shape
height = shape[0]
width = shape[1]

path = 'D:/data/sonics'
path = 'D:/data/sonics/augmented'


def create_training_data(data_path):
    training_data = []

    # iterate over each image
    for image in os.listdir(data_path):
        # check file extention
        if image.endswith(".jpg"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(str(full_name), 0)
                training_data.append([data])
            except Exception as err:
                print("an error has occured: ", err, str(full_name))

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, height, width)
    return training_data


data = create_training_data(path)
x_train = data[:-1]
x_test = data[-1:]
# check images
img = x_train[0]
plt.imshow(img, cmap='gray')

# build autoencoder


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # input layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(height*width, activation='sigmoid'),
            layers.Reshape((height, width))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(x_train, x_train,
                          epochs=40,
                          shuffle=True,
                          validation_data=(x_test, x_test))


# individual sample
# Load an image from a file
data = cv2.imread(str(file), 0)
normalized_data = data.astype('float32') / 255.
# test an image
encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
decoded = autoencoder.decoder(encoded)
loss = tf.keras.losses.mse(decoded, normalized_data)
sample_loss = np.mean(loss) + np.std(loss)
return sample_loss
