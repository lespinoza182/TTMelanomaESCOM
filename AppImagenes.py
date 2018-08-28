#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luisespinoza
"""
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np
import statistics
import time
import AnalisisDeImagenes

tiempoInicial = time.time()
rutaImagen = ("C:/Users/lespi/OneDrive/Documentos/TT/PruebasPython/LeerImagen/imagenes/lesion3.bmp")
imagen = Image.open(rutaImagen)
imagen.show()
imagenGris = imagen
imagenGris = imagen.convert('L')
imagenGris.show()
AnalisisDeImagenes.histograma(imagenGris)
'''Imagen a Escala de Grises
i = 0
while i < imagenGris.size[0]:
    j = 0
    while j < imagenGris.size[1]:
        r, g, b = imagenGris.getpixel((i,j))
        # Promedio de los colores
        gr = (r + g + b) / 3
        gris = int(gr)
        # Tupla de los 3 colores a gris
        pixel = tuple([gris, gris, gris])
        # Se intrudice la tupla como el pixel correspondiente
        imagenGris.putpixel((i,j), pixel)
        j += 1
    i += 1
imagenGris.show()'''
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
        #print('Y: ',y,'* width: ',width,' + x: ',x,' = ',loc)
        if imgAux[loc] > thr:
            imgOut[loc] = 255
        else:
            imgOut[loc] = 0
imgthr = imgOut
imOtsu = imgthr.reshape(height,width)
imOtsu = Image.fromarray(imOtsu)
imOtsu.show()
#AnalisisDeImagenes.histograma(imOtsu)
'''Máscara de la Imagen para quitar borde'''
centroImagen = [int(width/2), int(height/2)]
radio = min(centroImagen[0], centroImagen[1], width-centroImagen[0], height-centroImagen[1])
print('Centro de la imagen: ',centroImagen,'Radio: ',radio)
Y, X = np.ogrid[:height, :width]
distanciaCentro = np.sqrt((X - centroImagen[0])**2 + (Y - centroImagen[1])**2)
print('Y: ',y,'X: ',x,'Distancia del Centro: ',distanciaCentro)
mask = distanciaCentro <= radio
imMask = imOtsu.copy()
sh1, sh2 = np.shape(imMask)
#imMask = np.ndarray(shape=(sh1,sh2))
imMask = np.asarray(imMask, dtype = np.float32)
maskedImg = imMask.copy()
maskedImg[~mask] = 255
imgthr2 = maskedImg
imgSinBorde = imgthr2.reshape(height,width)
imgSinBorde = Image.fromarray(imgSinBorde)
imgSinBorde.show()
'''Morfología Matemática: Erosión'''
imgErosion = imgSinBorde
imgErosion = ndimage.grey_erosion(imgErosion, size=(5,5)).astype(np.float32)
imgErosion = imgErosion.reshape(height,width)
imgErosion = Image.fromarray(imgErosion)
imgErosion.show()
'''Morfología Matemática: Dilatación'''
imgDilatacion = imgSinBorde
imgDilatacion = ndimage.grey_dilation(imgDilatacion, size=(5,5)).astype(np.float32)
imgDilatacion = imgDilatacion.reshape(height,width)
imgDilatacion = Image.fromarray(imgDilatacion)
imgDilatacion.show()
'''Apertura'''
imgApertura = imgSinBorde
imgApertura = ndimage.grey_erosion(imgApertura, size=(5,5)).astype(np.float32)
imgApertura = imgApertura.reshape(height,width)
imgApertura = Image.fromarray(imgApertura)
imgApertura = ndimage.grey_dilation(imgApertura, size=(5,5)).astype(np.float32)
imgApertura = imgApertura.reshape(height,width)
imgApertura = Image.fromarray(imgApertura)
imgApertura.show()
'''Cerradura'''
imgCerradura = imgSinBorde
imgCerradura = ndimage.grey_dilation(imgCerradura, size=(5,5)).astype(np.float32)
imgCerradura = imgCerradura.reshape(height,width)
imgCerradura = Image.fromarray(imgCerradura)
imgCerradura = ndimage.grey_erosion(imgCerradura, size=(5,5)).astype(np.float32)
imgCerradura = imgCerradura.reshape(height,width)
imgCerradura = Image.fromarray(imgCerradura)
imgCerradura.show()
tiempoFinal = time.time()
tiempoTotal = tiempoFinal - tiempoInicial
print('EL tiempo total de ejecucion es: ',tiempoTotal)
