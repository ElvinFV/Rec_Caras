# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:16:14 2021

@author: Elvin Flores
"""

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import neurolab as nl
import random
from random import randrange 
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def Cargar_Foto(numper,numfot):
#     BDF = np.zeros( ( 22500, (numper * numfot) ) )
#     for persona in range(numper):
#         for foto in range(numfot):
#             root = "Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"
            

numPer = 3 # Numero de personas
numFot = 5 # Numero de fotos de cada persona
# BD1 = Cargar_Foto(numPer, numFot)
# Uno = np.ones((1, BD1.shape[1]))#Fila con cantidad de columnas de la base de datos
# Uno = Uno.astype('float32')
# Media = np.array([np.mean(BD1,axis=1)]) #Para que todos los vectores tengan el mismo ofset
# BDP = BD1 - (Media.T * Uno)
# ima=io.imread("Imagenes/S1/F1.bmp")
# plt.imshow(ima,cmap='gray')
datos_E=[]
# et=["E","E","E","E","E","A","A","A","A","A","J","J","J","J","J"]
et=[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
for persona in range(numPer):
       for foto in range(numFot):
           ima=np.reshape(io.imread("Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"),[150,150,1])
           ima=np.array(ima).astype(float)/255
           datos_E.append([ima])
# modeloD=tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape(150,150,1))
    
#     ])
# modeloD=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150,150,1)),
#   tf.keras.layers.Dense(150,activation='relu'),
#   tf.keras.layers.Dense(150,activation='relu'),
#   tf.keras.layers.Dense(1,activation='sigmoid')])
 
# modeloCNN = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),

#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(100, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(datos_E)

modeloCNN2 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
# modeloD.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])

# modeloCNN.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])

modeloCNN2.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

data_gen_entrenamiento = datagen.flow(datos_E, et, batch_size=32)

A = np.random.randint(1,4)
IPA = io.imread('Imagenes/S' + str(A) + '/F6.bmp') # Imagen de prueba

data_Test=datagen.flow(IPA)

