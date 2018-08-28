#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luisespinoza
"""
from PIL import Image
from matplotlib import pyplot as plt
#from collections import Counter
#from scipy import ndimage as ndi
#from skimage import feature
#import scipy.misc
import numpy as np
import statistics
#import random
#import scipy
import time
#import cv2

def abrirImagen(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    tiempoFinal = time.time()
    print('El Proceso abrirImagen tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    return imagen
    
def escalaDeGrises(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen2 = imagen
    i = 0
    while i < imagen2.size[0]:
        j = 0
        while j < imagen2.size[1]:
            r, g, b = imagen2.getpixel((i,j))
            # Promedio de los colores
            g = (r + g + b) / 3
            gris = int(g)
            pixel = tuple([gris, gris, gris])
            imagen2.putpixel((i,j), pixel)
            j += 1
        i += 1
    imagen2.show()
    tiempoFinal = time.time()
    print('El Proceso escalaDeGrises tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def maximoDeGrises(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen3 = imagen
    i = 0
    while i < imagen3.size[0]:
        j = 0
        while j < imagen3.size[1]:
            maximo = max(imagen3.getpixel((i,j)))
            pixel = tuple([maximo, maximo, maximo])
            imagen3.putpixel((i,j), pixel)
            j += 1
        i += 1
    print("El nivel maximo de gris es: ", maximo)
    imagen3.show()
    tiempoFinal = time.time()
    print('El Proceso maximoDegrises tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def minimoDeGrises(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen4 = imagen
    i = 0
    while i < imagen4.size[0]:
        j = 0
        while j < imagen4.size[1]:
            minimo = min(imagen4.getpixel((i,j)))
            pixel = tuple([minimo, minimo, minimo])
            imagen4.putpixel((i,j), pixel)
            j += 1
        i += 1
    print("El nivel minimo de gris es: ", minimo)
    imagen4.show()
    tiempoFinal = time.time()
    print('El Proceso minimoDeGrises tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def negativoColor(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen5 = imagen
    i = 0
    while i < imagen5.size[0]:
        j = 0
        while j < imagen5.size[1]:
            r, g, b = imagen5.getpixel((i,j))
            # Obtención de negativos de Rojo, Verde y Azul
            rn = 255 - r
            gn = 255 - g
            bn = 255 - b
            pixel = tuple([rn, gn, bn])
            imagen5.putpixel((i,j), pixel)
            j += 1
        i += 1
    imagen5.show()
    tiempoFinal = time.time()
    print('El Proceso negativoColor tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def negativoGrises(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen6 = imagen
    i = 0
    while i < imagen6.size[0]:
        j = 0
        while j < imagen6.size[1]:
            # Obtención de los grises
            gris1, gris2, gris3 = imagen6.getpixel((i,j))
            # Obtención de los negativos de los grises
            negativo1 = 255 - gris1
            negativo2 = 255 - gris2
            negativo3 = 255 - gris3
            pixel = tuple([negativo1, negativo2, negativo3])
            imagen6.putpixel((i,j), pixel)
            j += 1
        i += 1
    imagen6.show()
    tiempoFinal = time.time()
    print('El Proceso negativoGrises tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def blancoNegro(imagen, nivelGris):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen7 = imagen
    i = 0
    while i < imagen7.size[0]:
        j = 0
        while j < imagen7.size[1]:
            r, g, b = imagen7.getpixel((i,j))
            gris = (r + g + b) / 3
            if gris < nivelGris:
                imagen7.putpixel((i,j), (0,0,0))
            else:
                imagen7.putpixel((i,j), (255,255,255))
            j += 1
        i += 1
    imagen7.show()
    tiempoFinal = time.time()
    print('El Proceso blancoNegro tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def histograma(imagen):
    #tiempoInicial = time.time()
    #rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    #imagen = Image.open(rutaImagen)
    #imagen.show()
    #imagen = imagen.convert('L')
    imagen9 = imagen
    #imagen9.show()
    [ren, col] = imagen9.size
    total = ren * col
    a = np.asarray(imagen9, dtype = np.float32)
    a = a.reshape(1, total)
    a = a.astype(int)
    a = max(a)
    valor = 0
    maxd = max(a)
    grises = maxd
    vec = np.zeros(grises + 1)
    for i in range(total - 1):
        valor = a[i]
        vec[valor] = vec[valor] + 1
    plt.plot(vec)
    #tiempoFinal = time.time()
    #print('El Proceso histograma tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def brillo(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen10 = imagen
    arreglo = np.array(imagen10.size)
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < imagen10.size[0]:
        j = 0
        while j < imagen10.size[1]:
            suma = suma + imagen10.getpixel((i,j))
            j += 1
        i += 1
    brillo = suma / total
    brillo = int(brillo)
    print("El brillo de la imagen es: ", brillo)
    tiempoFinal = time.time()
    print('El Proceso brillo tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def contraste(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen11 = imagen
    arreglo = np.array(imagen11.size)
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < imagen11.size[0]:
        j = 0
        while j < imagen11.size[1]:
            suma = suma + imagen11.getpixel((i,j))
            j += 1
        i += 1
    brillo = suma / total
    i = 0
    while i < imagen11.size[0]:
        j = 0
        while j < imagen11.size[1]:
            aux = imagen11.getpixel((i,j)) - brillo
            suma = suma + aux
            j += 1
        i += 1
    cont = suma * suma
    cont = np.sqrt(suma / total)
    contraste = int(cont)
    print("El contraste de la imagen es: ", contraste)
    tiempoFinal = time.time()
    print('El Proceso contraste tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def suma(imagen, alpha):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen.show()
    imagen12 = imagen
    i = 0
    while i < imagen12.size[0]:
        j = 0
        while j < imagen12.size[1]:
            valor = imagen12.getpixel((i,j))
            valor =  valor + alpha
            if valor >= 255:
                valor = 255
            else:
                valor = valor
            imagen.putpixel((i,j),(valor))
            j += 1
        i += 1
    imagen12.show()
    tiempoFinal = time.time()
    print('El Proceso suma tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def resta(imagen, alpha):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen.show()
    imagen12 = imagen
    i = 0
    while i < imagen12.size[0]:
        j = 0
        while j < imagen12.size[1]:
            valor = imagen12.getpixel((i,j))
            valor =  valor - alpha
            if valor <= 0:
                valor = abs(valor)
            else:
                valor = valor
            imagen.putpixel((i,j),(valor))
            j += 1
        i += 1
    imagen12.show()
    tiempoFinal = time.time()
    print('El Proceso resta tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def multiplicacion(imagen, alpha):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen.show()
    imagen12 = imagen
    i = 0
    while i < imagen12.size[0]:
        j = 0
        while j < imagen12.size[1]:
            valor = imagen12.getpixel((i,j))
            valor =  valor * alpha
            if valor >= 255:
                valor = 255
            if valor <= 0:
                valor = valor
            imagen.putpixel((i,j),(valor))
            j += 1
        i += 1
    imagen12.show()
    tiempoFinal = time.time()
    print('El Proceso multiplicacion tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def division(imagen, alpha):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagen.show()
    imagen12 = imagen
    i = 0
    while i < imagen12.size[0]:
        j = 0
        while j < imagen12.size[1]:
            valor = imagen12.getpixel((i,j))
            valor =  valor / alpha
            valor = int(valor)
            if valor <= 0:
                valor = abs(valor)
            else:
                valor = valor
            imagen.putpixel((i,j),(valor))
            j += 1
        i += 1
    imagen12.show()
    tiempoFinal = time.time()
    print('El Proceso division tardo: ', tiempoFinal - tiempoInicial, 'Segundos')

def filtroMediana(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagenGris = imagen
    imagenGris.show()
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
            imagenGris.putpixel((i+1,j+1),(nuevo))
            j += 1
        i += 1
    imagenGris.show()
    tiempoFinal = time.time()
    print('El Proceso de Filtro de la Mediana tardo: ', tiempoFinal - tiempoInicial, 'Segundos')
    
def umbralDeOtsu(imagen):
    tiempoInicial = time.time()
    rutaImagen = ("/Users/luisespinoza/Desktop/TT/PruebasPython/LeerImagen/imagenes/" + imagen)
    imagen = Image.open(rutaImagen)
    imagen.show()
    imagen = imagen.convert('L')
    imagenGris = imagen
    imagenGris.show()
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
            m1 = media[1] / clase1
            m2 = (mt - media[i]) / clase2
            sigmaB2 = (clase1 * (m1 - mt) * (m1 - mt) + clase2 * (m2 - mt) * (m2 - mt))
            if sigmaB2 > sigmaB2max:
                sigmaB2max = sigmaB2
                T = i
    thr = int(T)
    print('El Umbral optimo de la imagen es: ',thr)
    # Aplicacion del metodo de Otsu
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
    tiempoFinal = time.time()
    print('El proceso de umbralDeOtsu tardo: ', tiempoFinal - tiempoInicial, 'Segundos')