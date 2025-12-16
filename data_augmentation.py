# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:36:46 2025

@author: Leonardo
"""

#%% Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#%% Diretório 
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)

import pathlib
data_dir = pathlib.Path(data_dir)
data_dir
list(data_dir.glob('*/*/*.jpg'))

image_count = len(list(data_dir.glob('*/*/*.jpg')))

roses = list(data_dir.glob('*/roses/*.jpg'))
roses[:5]

#%% Mostrar as imagens
# Rosas
PIL.Image.open(str(roses[0]))

# Tulipas
tulips = list(data_dir.glob('*/tulips/*.jpg'))
PIL.Image.open(str(tulips[0]))

#%% Dicionário dos diretórios
flowers_image_dict = {
    'roses': list(data_dir.glob('*/roses/*.jpg')),
    'daisy': list(data_dir.glob('*/daisy/*.jpg')),
    'dandelion': list(data_dir.glob('*/dandelion/*.jpg')),
    'sunflowers': list(data_dir.glob('*/sunflowers/*.jpg')),
    'tulips': list(data_dir.glob('*/tulips/*.jpg')) 
    }

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4 
    }

#%% Convertendo as fotos em array usando opencv
img = cv2.imread(str(flowers_image_dict['roses'][0]))
img.shape

cv2.resize(img, (180,180)).shape

X, y = [], []

for flower_name, images in flowers_image_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
        
X = np.array(X)
y = np.array(y)

#%% Separa a base em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Normalização
X_train_scaled = X_train/255
X_test_scaled = X_test/255

#%% Modelo CNN
num_classes = 5

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)

    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

model.evaluate(X_test_scaled, y_test)

# Predictions
predictions = model.predict(X_test_scaled)
predictions
score = tf.nn.softmax(predictions[0])
np.argmax(score)
y_test[0]

#%% Data Augmentation
img_height=180
img_width=180
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", 
                                                input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

])

# Original image
plt.axis('off')
plt.imshow(X[0])

# Newly generated training sample using data augmentation
plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


#%% Train the model using data augmentation and a drop out layer
num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)   


model.evaluate(X_test_scaled,y_test)
