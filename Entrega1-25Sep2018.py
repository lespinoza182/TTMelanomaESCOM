#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luisespinoza, manuelarreola
"""
from PIL import Image
from matplotlib import pyplot as plt
from numpy import array
from collections import Counter
import numpy as np
import statistics
import time
import math

tiempoInicial = time.time()
rutaImagen = ("C:/Users/lespi/OneDrive/Documentos/TT/PruebasPython/LeerImagen/imagenes/lesion1.bmp")
imagen = Image.open(rutaImagen)
imagen.show()
'''Imagen a Escala de Grises: Promedio'''
imgGrisProm = imagen
i = 0
while i < imgGrisProm.size[0]:
    j = 0
    while j < imgGrisProm.size[1]:
        r, g, b = imgGrisProm.getpixel((i, j))
        g = (r + g + b) / 3
        gris = int(g)
        pixel = tuple([gris, gris, gris])
        imgGrisProm.putpixel((i, j), pixel)
        j+=1
    i+=1
imgGrisProm.show()
'''Imagen a Escala de Grises: Método de Python'''
imgGris = imagen.convert('L')
imgGris.show()
'''Filtro Máximo de la Imagen'''
imgMax = imgGris
[ren, col] = imgMax.size
matriz = np.asarray(imgMax, dtype = np.float32)
i = 0
while i < ren - 3:
    j = 0
    while j < col - 3:
        submatriz = matriz[j:j+3,i:i+3]
        submatriz = submatriz.reshape(1, 9)
        nuevo = int(max(max(submatriz)))
        imgMax.putpixel((i + 1, j + 1),(nuevo))
        j+=1
    i+=1
imgMax.show()
'''Filtro Mínimo de la Imagen'''
imgMin = imgGris
[ren, col] = imgMin.size
matriz = np.asarray(imgMin, dtype = np.float32)
i = 0
while i < ren - 3:
    j = 0
    while j < col - 3:
        submatriz = matriz[j:j+3,i:i+3]
        submatriz = submatriz.reshape(1, 9)
        nuevo = int(min(min(submatriz)))
        imgMin.putpixel((i + 1, j + 1),(nuevo))
        j+=1
    i+=1
imgMin.show()
'''Filtro Moda de la Imagen'''
imgModa = imgGris
[ren, col] = imgModa.size
matriz = np.asarray(imgModa, dtype = np.float32)
i = 0
while i < ren - 3:
    j = 0
    while j < col - 3:
        submatriz = matriz[j:j+3,i:i+3]
        submatriz = submatriz.reshape(1, 9)
        nuevo = (max(submatriz))
        data = Counter(nuevo)
        nuevo = data.most_common(1)
        nuevo = max(nuevo)
        nuevo = int(nuevo[0])
        imgModa.putpixel((i + 1, j + 1),(nuevo))
        j+=1
    i+=1
imgModa.show()
'''Filtro Mediana de la Imagen'''
[ren, col] = imgGris.size
matriz = np.asarray(imgGris, dtype = np.float32)
i = 0
while i < ren - 3:
    j = 0
    while j < col - 3:
        submatriz = matriz[j:j+3,i:i+3]
        submatriz = submatriz.reshape(1,9)
        nuevo = (max(submatriz))
        nuevo = statistics.median(nuevo)
        nuevo = int(nuevo)
        imgGris.putpixel((i+1,j+1), (nuevo))
        j += 1
    i += 1
imagenMediana = imgGris
imagenMediana.show()
tiempoFin = time.time()
print('El Proceso Tardo: ', tiempoFin - tiempoInicial, ' Segundos')
