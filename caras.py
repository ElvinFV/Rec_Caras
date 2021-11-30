# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:55:52 2021

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

#-------------------------------- Cargar Imagen -----------------------------#
def Cargar_Foto(numper,numfot):
    BDF = np.zeros( ( 22500, (numper * numfot) ) )
    for persona in range(numper):
        for foto in range(numfot):
            root = "Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"
            Pic =io.imread(root)# Concatena las columnas en una sola
            Pic=np.ravel(np.array(Pic).astype(float)/255)
            BDF[:, ((numper * foto) + persona)] = Pic
    return BDF# Regresa matriz q|ade sujetos "base de datos"

#========================= Inicio de Face Recognition =======================#

#------------------------- Matriz de Base de Datos --------------------------#

numPer = 3 # Numero de personas
numFot = 5 # Numero de fotos de cada persona
BD1 = Cargar_Foto(numPer, numFot)
Uno = np.ones((1, BD1.shape[1]))#Fila con cantidad de columnas de la base de datos
Uno = Uno.astype('float32')
Media = np.array([np.mean(BD1,axis=1)]) #Para que todos los vectores tengan el mismo ofset
BDP = BD1 - (Media.T * Uno)

# #------------------------- EigenVectors y EigenValues -----------------------#

MC = np.dot((BDP.T),BDP)
#Eigenvalue,eigenfaces
[EV,EF] = np.linalg.eig(MC)
RE = np.dot(BDP,EF) # Matriz de caracteristicas
pr = 15 # Valores de los patrones principales
MR = np.array(RE[:, :-(1 + pr):-1])
#--------------------- Matriz de Caracteristicas Principales ----------------#

sign = np.zeros((BD1.shape[1], pr))
for i in range(0, BD1.shape[1]):
    sign[i, :] = np.dot(BDP[:, i].T, MR)
sign=sign/np.argmax(sign)
#------------------------- Selección con Red neuronal -------------------#
act=[0,0,0]
t=[0,0,0]
W_1=np.random.rand(1,15)
W_2=np.random.rand(1,15)
W_3=np.random.rand(1,15)
b_1=np.random.rand()
b_2=np.random.rand()
b_3=np.random.rand()
n_1=0
n_2=0
n_3=0
# P=np.transpose(sign[0,:])
# while(input()=='1'):
# for i in range(3):
#     if i==0: tar=[1,0,0]
#     if i==1: tar=[0,1,0]
#     if i==2: tar=[0,0,1]
for j in range(15):
    P=np.transpose(sign[j,:])
    if j<=0 and j<=4: tar=[1,0,0]
    if j<=5 and j<=9: tar=[0,1,0]
    if j<=10 and j<=14: tar=[0,0,1]
    # print("Imagen ",j)
    k=0
    cond=True
    while(k<500):      
        # print("Iteracion:",k)
        k=k+1
        #======= Primer Neurona======#
        
        n_1=np.dot(W_1,P)+b_1
        act[0]=1/(1 + np.exp(-n_1))
        # print("Neurona1:",act[0])
        e_1=tar[0]-act[0]
        W_1=W_1+e_1*sign[j,:]
        b_1=b_1+e_1
        #======= Segunda Neurona======#
        
        n_2=np.dot(W_2,P)+b_2
        act[1]=1/(1 + np.exp(-n_2))
        # print("Neurona2:",act[1])
        e_2=tar[1]-act[1]
        W_2=W_2+e_2*sign[j,:]
        b_2=b_2+e_2
        #======= Tercer Neurona======#
        
        n_3=np.dot(W_3,P)+b_3
        act[2]=1/(1 + np.exp(-n_3))
        # print("Neurona3:",act[2])
        e_3=tar[2]-act[2]
        W_3=W_3+e_3*sign[j,:]
        b_3=b_3+e_3
        # if j<=0 and j<=4 and act[0]>act[1] and act[0]>act[2]: cond=False
        # if j<=5 and j<=9 and act[1]>act[0] and act[1]>act[2]: cond=False
        # if j<=10 and j<=14 and act[2]>act[0] and act[2]>act[1]: cond=False
        
# act[0]=1/(1 + np.exp(-n_1))
# # print("Neurona1:",act[0])
# act[1]=1/(1 + np.exp(-n_2))
# # print("Neurona2:",act[1])
# act[2]=1/(1 + np.exp(-n_3))
# print("Neurona3:",act[2])

#------------------------- Prueba con selección Aleatoria -------------------#
A = np.random.randint(1,4)
IPA = io.imread('Imagenes/S' + str(A) + '/F6.bmp') # Imagen de prueba
Pic=np.ravel(np.array(IPA).astype(float)/255)

pro = Pic - Media

plt.subplot(121)
plt.imshow(np.reshape(Pic,[150,150]), cmap = 'gray');
plt.show(block=False)
plt.title('Buscando a...')

sal = np.dot(pro, MR)
sal=sal/np.argmax(sal)
sal=np.transpose(sal)

n_1=np.dot(W_1,sal)+b_1
act[0]=1/(1 + np.exp(-n_1))
print("Neurona1:",act[0])
n_2=np.dot(W_2,sal)+b_2
act[1]=1/(1 + np.exp(-n_2))
print("Neurona2:",act[1])
n_3=np.dot(W_3,sal)+b_3
act[2]=1/(1 + np.exp(-n_3))
print("Neurona3:",act[2])
# MS = np.zeros(pr)

plt.subplot(122)
#------------------------- Selección con Distancia Euclidiana -------------------#
# for i in range(BD1.shape[1]):
#     MS[i] = np.linalg.norm(sign[i,:] - sal) # Distancia Euclidiana
#     print("Norma",MS[i])
#     if (np.remainder(i, 2) == 0):
#         plt.imshow(np.reshape(BD1[:, i], [150,150]), cmap = 'gray');
#         plt.show(block=False)
#         plt.pause(2)
suj = np.argmax(act)
C=0
if suj==0:C=np.random.randint(0,5)
if suj==1:C=np.random.randint(5,10)
if suj==2:C=np.random.randint(10,15)
#------------------------- Ploteo Imagenes -------------------#
plt.imshow(np.reshape(BD1[:,C],[150,150]), cmap = 'gray');
plt.title('¡Se ha encontrado!')
plt.show()


# # def Cargar_Foto(numper,numfot):
# #     BDF = np.zeros( ( 22500, (numper * numfot) ) )
# #     for persona in range(numper):
# #         for foto in range(numfot):
# #             root = "Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"
            

# numPer = 3 # Numero de personas
# numFot = 5 # Numero de fotos de cada persona
# # BD1 = Cargar_Foto(numPer, numFot)
# # Uno = np.ones((1, BD1.shape[1]))#Fila con cantidad de columnas de la base de datos
# # Uno = Uno.astype('float32')
# # Media = np.array([np.mean(BD1,axis=1)]) #Para que todos los vectores tengan el mismo ofset
# # BDP = BD1 - (Media.T * Uno)
# ima=io.imread("Imagenes/S1/F1.bmp")
# plt.imshow(ima,cmap='gray')
# datos_E=[]
# # et=["E","E","E","E","E","A","A","A","A","A","J","J","J","J","J"]
# et=[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
# for persona in range(numPer):
#        for foto in range(numFot):
#            ima=np.reshape(io.imread("Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"),[150,150,1])
#            ima=np.array(ima).astype(float)/255
#            datos_E.append([ima])
# # modeloD=tf.keras.models.Sequential([
# #     tf.keras.layers.Flatten(input_shape(150,150,1))
    
# #     ])
# # modeloD=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150,150,1)),
# #   tf.keras.layers.Dense(150,activation='relu'),
# #   tf.keras.layers.Dense(150,activation='relu'),
# #   tf.keras.layers.Dense(1,activation='sigmoid')])
 
# # modeloCNN = tf.keras.models.Sequential([
# #   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
# #   tf.keras.layers.MaxPooling2D(2, 2),
# #   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# #   tf.keras.layers.MaxPooling2D(2, 2),
# #   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
# #   tf.keras.layers.MaxPooling2D(2, 2),

# #   tf.keras.layers.Flatten(),
# #   tf.keras.layers.Dense(100, activation='relu'),
# #   tf.keras.layers.Dense(1, activation='sigmoid')
# # ])
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=15,
#     zoom_range=[0.7, 1.4],
#     horizontal_flip=True,
#     vertical_flip=True
# )

# datagen.fit(datos_E)

# modeloCNN2 = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),

#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(250, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# # modeloD.compile(optimizer='adam',
# #                     loss='binary_crossentropy',
# #                     metrics=['accuracy'])

# # modeloCNN.compile(optimizer='adam',
# #                     loss='binary_crossentropy',
# #                     metrics=['accuracy'])

# modeloCNN2.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])

# data_gen_entrenamiento = datagen.flow(datos_E, et, batch_size=32)