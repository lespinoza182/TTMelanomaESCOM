#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luisespinoza, manuelarreola
"""
from PIL import Image
from matplotlib import pyplot as plt
from numpy import array
import numpy as np
import statistics
import time
import math

import AnalisisDeImagenes

tiempoInicial = time.time()
rutaImagen = ("C:/Users/lespi/OneDrive/Documentos/TT/PruebasPython/LeerImagen/imagenes/lesion1.bmp")
imagen = Image.open(rutaImagen)
imagen.show()
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
tiempoFinal = time.time()
tiempoTotal = tiempoFinal - tiempoInicial
print('El tiempo total de ejecucion es: ',tiempoTotal)
