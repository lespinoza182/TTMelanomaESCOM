#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luisespinoza, manuelarreola
"""
from PIL import Image
from matplotlib import pyplot as plt
from numpy import array
from collections import Counter
import functools
import numpy as np
import statistics
import time
import math

import AnalisisDeImagenes

tiempoInicial = time.time()
rutaImagen = ("/home/lespinoza182/Documentos/TTMelanomaESCOM-master/imagenes/lesion1.bmp")
imagen = Image.open(rutaImagen)
imagen.show()
'''ANALISIS DE LA IMAGEN'''
imagenGris = imagen
imagenGris = imagen.convert('L')
imagenGris.show()
AnalisisDeImagenes.histograma(imagenGris)
'''Filtro de la Mediana'''
[ren, col] = imagenGris.size
matriz = np.asarray(imagenGris, dtype = np.float32)
i = 0
while i < ren - 3:
    j = 0
    while j < col - 3:
        submatriz = matriz[j:j+3,i:i+3]
        submatriz = submatriz.reshape(1,9)
        nuevo = (max(submatriz))
        nuevo = statistics.median(nuevo)
        nuevo = int(nuevo)
        imagenGris.putpixel((i+1,j+1), (nuevo))
        j += 1
    i += 1
imagenMediana = imagenGris
imagenMediana.show()
#AnalisisDeImagenes.histograma(imagenMediana)
'''Umbral Optimo de Otsu'''
width, height = imagenGris.size
imgAux = np.array(imagenGris.getdata())
histogram = np.array(imagenGris.histogram(),float) / (width * height)
# Probabilidad acumulada
omega = np.zeros(256)
# Media acumulada
media = np.zeros(256)
# Calculo de probabilidad acumulada y media acumulada a partir de histograma
omega[0] = histogram[0]
for i in range(len(histogram)):
    omega[i] = omega[i - 1] + histogram[i]
    media[i] = media[i - 1] + (i - 1) * histogram[i]
sigmaB2 = 0
mt = media[len(histogram) - 1]
sigmaB2max = 0
T = 0
for i in range(len(histogram)):
    clase1 = omega[i]
    clase2 = 1 - clase1
    if clase1 != 0 and clase2 != 0:
        m1 = media[i] / clase1
        m2 = (mt - media[i]) / clase2
        sigmaB2 = (clase1 * (m1 - mt) * (m1 - mt) + clase2 * (m2 - mt) * (m2 - mt))
        if sigmaB2 > sigmaB2max:
            sigmaB2max = sigmaB2
            T = i
thr = int(T)
print('El Umbral optimo de la imagen es: ',thr)
'''Metodo de Otsu'''
xmin, xmax = 0, width
ymin, ymax = 0, height
imgOut = np.zeros(width * height)
loc = 0
for y in range(ymin, ymax):
    for x in range(xmin, xmax):
        loc = y * width + x
        if imgAux[loc] > thr:
            imgOut[loc] = 255
        else:
            imgOut[loc] = 0
imgthr = imgOut
imOtsu = imgthr.reshape(height,width)
imOtsu = Image.fromarray(imOtsu)
imOtsu.show()
#AnalisisDeImagenes.histograma(imOtsu)

#Máscara de la Imagen para quitar borde
centroImagen = [int(width/2), int(height/2)]
radio = min(centroImagen[0], centroImagen[1], width-centroImagen[0], height-centroImagen[1])
print('Centro de la imagen: ',centroImagen,'Radio: ',radio)
Y, X = np.ogrid[:height, :width]
distanciaCentro = np.sqrt((X - centroImagen[0])**2 + (Y - centroImagen[1])**2)
#print('Y: ',y,'X: ',x,'Distancia del Centro: ',distanciaCentro)
mask = distanciaCentro <= radio
imMask = imOtsu.copy()
sh1, sh2 = np.shape(imMask)
imMask = np.asarray(imMask, dtype = np.float32)
maskedImg = imMask.copy()
maskedImg[~mask] = 255
imgthr2 = maskedImg
imgSinBorde = imgthr2.reshape(height,width)
imgSinBorde = Image.fromarray(imgSinBorde)
imgSinBorde.show()

'''Morfología Matemática: Apertura'''
#Erosión
a = np.asarray(imgSinBorde, dtype = np.float32)
#Es necesario aumentar el rango para más ciclos
for z in range(1, 4):
    xmin = 1
    xmax = height - 1
    ymin = 1
    ymax = width - 1
    imgErosion = np.zeros((height,width), dtype = np.float32)
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            aux = max(a[x,y], a[x-1,y], a[x+1,y], a[x,y-1], a[x,y+1])
            imgErosion[x,y] = aux
    a = imgErosion
imgErosion = a
imgErosion = imgErosion.reshape(height,width)
imgErosion = Image.fromarray(imgErosion)
imgErosion.show()
#Dilatación
b = np.asarray(imgErosion, dtype = np.float32)
#Es necesario aumentar el rango para más ciclos
for z in range(1, 4):
    xmin = 1
    xmax = height - 1
    ymin = 1
    ymax = width - 1
    imgDilatacion = np.zeros((height,width), dtype = np.float32)
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            aux = min(b[x,y], b[x-1,y], b[x+1,y], b[x,y-1], b[x,y+1])
            imgDilatacion[x,y] = aux
    b = imgDilatacion
imgDilatacion = b
imgDilatacion = imgDilatacion.reshape(height,width)
imgDilatacion = Image.fromarray(imgDilatacion)
imgApertura = imgDilatacion
imgApertura.show()
'''Detección del Borde: Sobel'''
sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
sobel_x = np.array(sobel_x)
sobel_y = np.array(sobel_y)
imgSobel = np.asarray(imgApertura, dtype = np.float64)
imgBorde = np.zeros((height,width), dtype = np.float64)
xmin = 1
xmax = height - 1
ymin = 1
ymax = width - 1
for x in range(xmin, xmax):
    for y in range(ymin, ymax):
        pixel_x = (sobel_x[0,0]*imgSobel[x-1,y-1]) + (sobel_x[0,1]*imgSobel[x,y-1]) + (sobel_x[0,2]*imgSobel[x+1,y-1])\
        + (sobel_x[1,0]*imgSobel[x-1,y]) + (sobel_x[1,1]*imgSobel[x,y]) + (sobel_x[1,2]*imgSobel[x+1,y])\
        + (sobel_x[2,0]*imgSobel[x-1,y+1]) + (sobel_x[2,1]*imgSobel[x,y+1]) + (sobel_x[2,2]*imgSobel[x+1,y+1])
        pixel_y = (sobel_y[0,0]*imgSobel[x-1,y-1]) + (sobel_y[0,1]*imgSobel[x,y-1]) + (sobel_y[0,2]*imgSobel[x+1,y-1])\
        + (sobel_y[1,0]*imgSobel[x-1,y]) + (sobel_y[1,1]*imgSobel[x,y]) + (sobel_y[1,2]*imgSobel[x+1,y])\
        + (sobel_y[2,0]*imgSobel[x-1,y+1]) + (sobel_y[2,1]*imgSobel[x,y+1]) + (sobel_y[2,2]*imgSobel[x+1,y+1])
        val = math.sqrt((pixel_x*pixel_x)+(pixel_y*pixel_y))
        imgBorde[x,y] = val
imgBorde = imgBorde.reshape(height,width)
imgBorde = Image.fromarray(imgBorde)
imgBorde.show()
'''Negativo Imagen de Sobel'''
imgBordeNeg = imgBorde
i = 0
while i < imgBordeNeg.size[0]:
    j = 0
    while j < imgBordeNeg.size[1]:
        gris = imgBordeNeg.getpixel((i,j))
        valor = 255 - gris
        imgBordeNeg.putpixel((i, j), valor)
        j+=1
    i+=1
imgBordeNeg.show()
'''Negativo Imagen Apertura'''
imgAperturaNeg = imgApertura
i = 0
while i < imgAperturaNeg.size[0]:
    j = 0
    while j < imgAperturaNeg.size[1]:
        gris = imgAperturaNeg.getpixel((i,j))
        valor = 255 - gris
        imgAperturaNeg.putpixel((i, j),valor)
        j+=1
    i+=1
imgAperturaNeg.show()
'''Suma de las Imágenes'''
imgFinal = imagen
i  = 0
while i < imgAperturaNeg.size[0]:
    j = 0
    while j < imgAperturaNeg.size[1]:
        if imgAperturaNeg.getpixel((i,j)) == 0:
            imgFinal.putpixel((i,j),0)
        j+=1
    i+=1
imgFinal.show()
imgFinal = imagenGris
i  = 0
while i < imgAperturaNeg.size[0]:
    j = 0
    while j < imgAperturaNeg.size[1]:
        if imgAperturaNeg.getpixel((i,j)) == 0:
            imgFinal.putpixel((i,j),0)
        j+=1
    i+=1
#imgFinal.show()
'''RECONOCIMIENTO DE PATRONES'''
'''Matriz de Co-Ocurrencia'''
[row, col]  = imgFinal.size
imgCo = np.asarray(imgFinal, dtype = np.int)
print(imgCo)
histogram = np.array(imgFinal.histogram(),int) / (width * height)
nivelesGris = len(Counter(histogram))
print("Niv.Gris: ",nivelesGris,"Row: ",row,"Col: ")
datos = imgCo
datos=np.asarray(datos)
[ren, col] = datos.shape
total = ren * col
nm = datos.reshape((1, total))
nm = max(nm)
x=max(nm)
"""0º Grados"""
print("-----0º Grados-----")
cero = np.zeros((x + 1, x + 1))
cont = 1
i = 0
while i < (total - 1):
    n1 = nm[i]
    n2 = nm[i + 1]
    cero[n1, n2] = cero[n1, n2] + 1
    cero[n2, n1] = cero[n2, n1] + 1
    if(cont == (ren - 1)):
        i = i + 2
        cont = 1
    else:
        i = i + 1
        cont = cont + 1
print(cero)
[ren, col]  = cero.shape
entropy  = energy = contrast = homogeneity = disimility = None
prob_ac = 0
normalizer = functools.reduce(lambda x,y: x + sum(y), cero, 0)
print("Ren: ",ren,"Col: ",col,"Normalizador: ",normalizer)
for m in range(col):
    for n in range(ren):
        prob = (1.0 * cero[m][n]) / normalizer
        if prob > prob_ac:
            prob_ac = prob
        if (prob >= 0.0001) and (prob <= 0.999):
            log_prob = math.log(prob,2)
        if prob < 0.0001:
            log_prob = 0
        if prob > 0.999:
            log_prob = 0
        if entropy is None:
            entropy = -1.0 * prob * log_prob
            continue
        entropy += -1.0 * prob * log_prob #Checked
        if energy is None:
            energy = prob ** 2
            continue
        energy += prob ** 2
        if contrast is None: #Checked
            contrast = ((m - n)**2) * prob
            continue
        contrast += ((m - n)**2) * prob
        if homogeneity is None:
            homogeneity = prob / ((1 + abs(m - n))*1.0)
            continue
        homogeneity += prob / ((1 + abs(m - n))*1.0) #Checked
        if disimility is None:
            disimility = prob * abs(m - n)
            continue
        disimility += prob * abs(m - n)
if abs(entropy) < 0.0000001: entropy = 0.0
print("\nVector de Características de la imagen: ")
print("[Entropy: ",entropy)
print(", Contrast: ",contrast)
print(", Homogeneity: ",homogeneity)
print(", Energy: ",energy)
print(", Maximum Probability: ",prob_ac)
print(", Disimility: ",disimility)
print("]");
tiempoFinal = time.time()
tiempoTotal = tiempoFinal - tiempoInicial
print('El tiempo total de ejecucion es: ',tiempoTotal)
