# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:35:46 2021

@author: Elvin Flores
"""

import numpy as np
from skimage import io, color
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils


numPer = 3 # Numero de personas
numFot = 5 # Numero de fotos de cada persona

datos_T=[]

# et=["E","E","E","E","E","A","A","A","A","A","J","J","J","J","J"]
etiquetas_T=[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
for persona in range(numPer):
       for foto in range(numFot):
           ima=np.reshape(io.imread("Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"),[150,150,1])
           ima=np.array(ima).astype(float)/255
           datos_T.append([ima])

datos_Test=[]
etiquetas_Test=[0,0,0.5,0.5,1,1]
for persona in range(numPer):
       for foto in range(4,6):
           ima=np.reshape(io.imread("Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"),[150,150,1])
           ima=np.array(ima).astype(float)/255
           datos_Test.append([ima])
y_train = np_utils.to_categorical(etiquetas_T)
y_test = np_utils.to_categorical(etiquetas_Test)
class_num =3

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=np.shape(datos_T)[2:], padding='same'))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(class_num))
model.add(Activation('softmax'))

epochs = 25
optimizer="adam"
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
# print(model.summary())

scores = model.evaluate(datos_Test,etiquetas_Test, verbose=0)