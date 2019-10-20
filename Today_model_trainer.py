import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow import keras
import subprocess
import tempfile
import time
from glob import glob
from keras import preprocessing
from keras.utils import to_categorical

class_names = ['S1', 'S2', 'S3']

ImgSize = 80
width = 80
height = 80

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

images_type_1 = load_images('./data2/s1')
images_type_2 = load_images('./data2/s2')
images_type_3 = load_images('./data2/s3')

X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)

print(X_type_1.shape)
print(X_type_2.shape)
print(X_type_3.shape)

X = np.concatenate((X_type_1, X_type_2, X_type_3), axis=0)
X = X / 255.0
print(X.shape)

y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [1 for item in enumerate(X_type_2)]
y_type_3 = [2 for item in enumerate(X_type_3)]

y = np.concatenate((y_type_1, y_type_2, y_type_3), axis=0)
y = to_categorical(y, num_classes=len(class_names))
print(y.shape)


model = keras.Sequential([
  keras.layers.Conv2D(32, (3, 3), input_shape=(width, height, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Dropout(0.25),

  keras.layers.Conv2D(64, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2,2)),
  keras.layers.Dropout(0.25),

  keras.layers.Flatten(),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dropout(0.5),
  # keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
  keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 15
model.fit(X, y, epochs=epochs)
model.save('S2.h5')