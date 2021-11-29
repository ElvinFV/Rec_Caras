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


#-------------------------------- Cargar Imagen -----------------------------#
def Cargar_Foto(numper,numfot):
    BDF = np.zeros( ( 22500, (numper * numfot) ) )
    for persona in range(numper):
        for foto in range(numfot):
            root = "Imagenes/S" + str(persona + 1) + "/" + "F" + str(foto + 1) + ".bmp"
            Pic = np.ravel(io.imread(root))# Concatena las columnas en una sola
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

#------------------------- Selección con Red neuronal -------------------#
act=[0,0,0]
t=[0,0,0]
W_1=np.random.rand(1,15)
W_2=np.random.rand(1,15)
W_3=np.random.rand(1,15)
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
    if j==0: tar=[1,0,0]
    if j==5: tar=[0,1,0]
    if j==10: tar=[0,0,1]
    while(tar[0]==t[0] and tar[1]==t[1] and tar[2]==t[2]):
        #======= Primer Neurona======#
        b_1=np.random.rand()
        n_1=np.dot(W_1,P)+b_1
        act[0]=1/(1 + np.exp(-n_1))
        print("Neurona1:",act[0])
        e_1=t[0]-act[0]
        W_1=W_1+e_1[0]
        b_1=b_1+e_1
        #======= Segunda Neurona======#
        b_2=np.random.rand()
        n_2=np.dot(W_2,P)+b_2
        act[1]=1/(1 + np.exp(-n_2))
        print("Neurona2:",act[2])
        e_2=t[1]-act[1]
        W_2=W_2+e_2[0]
        b_2=b_2+e_2
        #======= Tercer Neurona======#
        b_3=np.random.rand()
        n_3=np.dot(W_3,P)+b_3
        act[2]=1/(1 + np.exp(-n_3))
        print("Neurona3:",act[2])
        e_3=t[2]-act[2]
        W_3=W_3+e_3[0]
        b_3=b_3+e_3
act[0]=1/(1 + np.exp(-n_1))
print("Neurona1:",act[0])
act[1]=1/(1 + np.exp(-n_2))
print("Neurona2:",act[2])
act[2]=1/(1 + np.exp(-n_3))
print("Neurona3:",act[2])

#------------------------- Prueba con selección Aleatoria -------------------#
A = np.random.randint(1,4)
IPA = io.imread('Imagenes/S' + str(A) + '/F6.bmp') # Imagen de prueba
FA = np.ravel(IPA)
pro = FA - Media

plt.subplot(121)
plt.imshow(np.reshape(FA,[150,150]), cmap = 'gray');
plt.show(block=False)
plt.title('Buscando a...')

sal = np.dot(pro, MR)
MS = np.zeros(pr)

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