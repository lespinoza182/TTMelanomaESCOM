# -*- coding: utf-8 -*-

from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from scipy import ndimage as ndi
from skimage import feature
import scipy.misc
import numpy as np
import statistics
import random
import scipy
import time
#import cv2


"""ABRIR UNA IMAGEN A COLOR Y A GRISES"""
def abrir_imagen(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""ESCALA DE GRISES DE LA IMAGEN A COLOR"""
def escala_de_grises(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im2 = im
    i = 0
    while i < im2.size[0]:
        j = 0
        while j < im2.size[1]:
            r, g, b = im2.getpixel((i, j))
            g = (r + g + b) / 3
            gris = int(g)
            pixel = tuple([gris, gris, gris])
            im2.putpixel((i, j), pixel)
            j+=1
        i+=1
    im2.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""MAXIMO DE GRISES DE LA IMAGEN A COLOR"""
def maximo(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im3 = im
    i = 0
    while i < im3.size[0]:
        j = 0
        while j < im3.size[1]:
            maximo = max(im3.getpixel((i, j)))
            pixel = tuple([maximo, maximo, maximo])
            im3.putpixel((i, j), pixel)
            j+=1
        i+=1
    print("El Valor Maximo De Grises Es: ", maximo)
    im3.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""MINIMO DE GRISES DE LA IMAGEN A COLOR"""
def minimo(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im4 = im
    i = 0
    while i < im4.size[0]:
        j = 0 
        while j < im4.size[1]:
            minimo = min(im4.getpixel((i, j)))
            pixel = tuple([minimo, minimo, minimo])
            im4.putpixel((i, j), pixel)
            j+=1
        i+=1
    print("El Valor Maximo De Grises Es: ", minimo)
    im4.show() 
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""NEGATIVO DE LA IMAGEN A COLOR"""
def negativo_color(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im5 = im
    i = 0
    while i < im5.size[0]:
        j = 0
        while j < im5.size[1]:
            r, g, b = im5.getpixel((i, j))
            rn = 255 - r
            gn = 255 - g
            bn = 255 - b
            pixel = tuple([rn, gn, bn])
            im5.putpixel((i, j), pixel)
            j+=1
        i+=1
    im5.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
   
"""NEGATIVO DE LA IMAGEN A GRISES"""
def negativo_grises(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im15 = im
    i = 0
    while i < im15.size[0]:
        j = 0
        while j < im15.size[1]:
            gris = im15.getpixel((i,j))
            valor = 255 - gris
            im15.putpixel((i, j), valor)
            j+=1
        i+=1
    im15.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
   
   
"""BLANCO Y NEGRO DE LA IMAGEN A COLOR"""
def blanco_negro(im,grisBase):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im6 = im
    i = 0
    while i < im6.size[0]:
        j = 0
        while j < im6.size[1]:
            r, g, b = im6.getpixel((i, j))
            gris = (r + g + b) / 3
            if gris < grisBase:
                im6.putpixel((i, j), (0, 0, 0))
            else:
                im6.putpixel((i, j), (255, 255, 255))   
            j+=1
        i+=1
    im6.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')    
    
    
"""TRAMSPUESTA DE LA IMAGEN A GRISES"""
def tramspuesta(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im7 = im
    ar = np.zeros((im7.size[0], im7.size[1]))
    i = 0 
    while i < im7.size[1]:
        j = 0
        while j < im7.size[0]:
            a = im7.getpixel((j, i))
            ar[j, i] = a
            j+=1
        i+=1
    ar = ar.astype(int)    
    im7 = Image.fromarray(ar)
    im7.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')    
    

"""HISTOGRAMA DE LA IMAGEN A COLOR"""
def histograma_color(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im8 = im
    arregloim8 = np.asarray(im8)
    plt.subplot(221), plt.imshow(im8)
    color = ('r','g','b')
    for i,col in enumerate(color):
        histr = cv2.calcHist([arregloim8],[i],None,[256],[0,256])
        plt.subplot(222), plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.xlim([0,256])
    plt.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""HISTOGRAMA DE LA IMAGEN A GRISES"""
def histograma_grises(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im16 = im
    [ren, col] = im16.size
    total = ren * col
    a = np.asarray(im16, dtype = np.float32)
    a = a.reshape(1, total)
    a = a.astype(int)
    a = max(a)
    valor = 0
    maxd = max(a)
    grises = maxd
    vec=np.zeros(grises + 1)
    for i in range(total - 1):
        valor = a[i]
        vec[valor] = vec[valor] + 1
    plt.plot(vec)
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""BRILLO DE LA IMAGEN A GRISES"""
def brillo(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im9 = im
    arreglo = np.array(im9.size)  
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < im9.size[0]:
        j = 0
        while j < im9.size[1]:
            suma = suma + im9.getpixel((i, j))
            j+=1
        i+=1
    brillo = suma / total    
    print("El brillo de la imagen es", brillo)  
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""CONTRASTE DE LA IMAGEN A GRISES"""
def contraste(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im10 = im
    arreglo = np.array(im10.size)  
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < im10.size[0]:
        j = 0
        while j < im10.size[1]:
            suma = suma + im10.getpixel((i, j))
            j+=1
        i+=1
    brillo = suma / total
    i = 0 
    while i < im10.size[0]:
        j = 0
        while j < im10.size[1]:
            aux = im10.getpixel((i,j)) - brillo 
            suma = suma + aux
            j+=1
        i+=1
    cont = suma * suma
    cont = np.sqrt(suma / total)
    contraste = int(cont)
    print("El contraste de la imagen es", contraste) 
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""SUMA DE GRISES EN LA IMAGEN A GRISES"""
def suma(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im11 = im
    i = 0
    while i < im11.size[0]:
        j = 0
        while j < im11.size[1]:
            valor = im11.getpixel((i, j))
            valor = valor + alpha
            if valor >= 255:
                valor = 255
            else:
                valor = valor
            im11.putpixel((i, j),(valor))
            j+=1
        i+=1
    im11.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')    
            

"""RESTA DE GRISES EN LA IMAGEN A GRISES"""
def resta(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im12 = im
    i = 0
    while i < im12.size[0]:
        j = 0
        while j < im12.size[1]:
            valor = im12.getpixel((i, j)) 
            valor = valor - alpha
            if valor <= 0:
                valor = abs(valor)
            else:
                valor = valor
            im12.putpixel((i, j),(valor))
            j+=1
        i+=1
    im12.show()   
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""MULTIPLICACION DE GRISES EN LA IMAGEN A GRISES"""
def multiplicacion(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im13 = im
    i = 0
    while i < im13.size[0]:
        j = 0
        while j < im13.size[1]:
            valor = im13.getpixel((i, j)) 
            valor = valor * alpha
            if valor >= 255:
                valor = 255
            if valor <= 0:
                valor = valor
            im13.putpixel((i, j),(valor))
            j+=1
        i+=1
    im13.show() 
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
                

"""DIVISION DE GRISES EN LA IMAGEN A GRISES"""
def division(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    im14 = im
    i = 0
    while i < im14.size[0]:
        j = 0
        while j < im14.size[1]:
            valor = im14.getpixel((i, j)) 
            valor = valor / alpha
            valor = int(valor)
            if valor <= 0:
                valor = abs(valor)
            else:
                valor = valor
            im14.putpixel((i, j),(valor))
            j+=1
        i+=1
    im14.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""SUMA DE DOS IMAGENES A GRISES"""
def suma_imagenes(im):
    tiempoIn = time.time()
    imagen1 = Image.open("C:/Users/CkriZz/Pictures/1.jpeg")
    imagen2 = Image.open("C:/Users/CkriZz/Pictures/2.jpeg")
    imagen1.show()
    imagen2.show()
#   para realizar blending deben tener el mismo tamano
    imagen1.resize(imagen2.size)
#   out = image1 * (1.0 - alpha) + image2 * alpha
#   alpha * imagen1 + (1.0 - alpha) * imagen2
    out = Image.blend(imagen1, imagen2, 0.50)
    out.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')



"""CONVOLUCION DE LA IMAGEN A GRISES"""
def convolucion(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    print("Dimesion De La Matriz:")
    dimension = input()
    dimension = int(dimension)
    datos = []
    i = 0
    print("Datos De La Matriz: ")
    while i < dimension:
        j = 0
        while j < dimension:
            nuevo = input()
            nuevo = float(nuevo)
            datos.append(nuevo)
            j+=1
        i+=1
    datos = np.asarray(datos, dtype = np.float32)
    datos = datos.reshape(dimension, dimension)
    [col,ren] = ima.size
    imagen1 = np.asarray(ima, dtype = np.float32)
    imagen2 = imagen1
    i = 0
    while i < ren-dimension:     
        j = 0
        while j < col-dimension:
            sub = imagen1[i:(dimension + i), j:(dimension + j)]
            suma = 0
            r = 0
            while r < dimension:     
                c=0
                while c < dimension:
                    suma = suma + sub[r,c] * datos[r,c]
                    c+=1
                r+=1
            valor = suma / (dimension * dimension)
            indice1 = ((dimension / 2 + .5) + i)
            indice2 = ((dimension / 2 + .5) + j)
            imagen2[indice1, indice2]=valor
            j+=1
        i+=1
    imagen2=Image.fromarray(imagen2)
    imagen2.show()  
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""ECUALIZACION NORMAL DE LA IMAGEN A GRISES"""
def ecua_normal(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j<col:
            valor = ima[j, i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo 
    mapeo = np.floor(mindata * ac)
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')    
    

"""ECUALIZACION UNIFORME DE LA IMAGEN A GRISES"""
def ecua_uniforme(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j<col:
            valor = ima[j, i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo 
    m1 = maxdata - mindata
    m2 = m1 * ac
    m3 = m2 + mindata
    mapeo = np.floor(m3)
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    

"""ECUALIZACION EXPONENCIAL DE LA IMAGEN A GRISES"""
def ecua_exponencial(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j < col:
            valor = ima[j,i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo
    m1 = 1 - ac
    mapeo = np.floor(mindata - 1 / alpha * np.log(m1)) 
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    

"""ECUALIZACION RAYLEIGH DE LA IMAGEN A GRISES"""
def ecua_rayleigh(im,alpha):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j < col:
            valor = ima[j,i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo
    m1 = 1 - ac
    m2 = 1 / m1    
    m3 = alpha * alpha    
    m4 = 2 * m3
    m5 = np.log(m2)
    m6 = m4 * m5
    m7 = pow(m6, 1/2)
    m8 = mindata + m7
    mapeo = np.floor(m8)
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""ECUALIZACION RAYLEIGH DE LA IMAGEN A GRISES"""
def ecua_hypercubica(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j < col:
            valor = ima[j,i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo
    m1 = pow(maxdata, 1/3)
    m2 = pow(mindata, 1/3)
    m3 = m2 * ac
    m4 = m1 - m3
    m5 = m4 + m1
    m6 = pow(m5 , 3)
    mapeo = np.floor(m6)
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    

"""ECUALIZACION NHYPERBOLICA DE LA IMAGEN A GRISES"""
def ecua_hyperbolica(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size 
    ima = np.asarray(ima, dtype = np.float32).reshape(1, ren * col)
    valor = 0 
    maxdata = max(max(ima))
    mindata = min(min(ima))
    niveles = maxdata
    h = np.zeros(niveles)
    ima = ima.reshape(col, ren)
    ac = h
    i = 0
    #cálculo del histograma
    while i < ren:
        j = 0
        while j < col:
            valor = ima[j, i] - 1
            h[valor] = h[valor] + 1
            j+=1
        i+=1
    ac[0] = h[0]
    i = 1
    while i < maxdata:
        ac[i] = ac[i - 1] + h[i]
        i+=1
    ac = ac / (ren * col)
    #funcion de mapeo
    m1 = maxdata / mindata
    m2 = mindata * m1
    m3 = pow(m2, ac)
    mapeo = np.floor(m3) 
    #si mindata es cero la imagen sera cero
    newim = np.zeros((col, ren))
    i = 0
    while i < ren:
        j = 0
        while j < col:
            newim[j, i] = mapeo[ima[j, i] - 1]
            j+=1
        i+=1
    newim = Image.fromarray(newim)
    newim.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""RUIDO GAUSSIANO EN LA IMAGEN A GRISES """
def ruido_gaussiano(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    l = scipy.misc.imread(ruta)
    noisy = l + 0.4 * l.std() * np.random.random(l.shape)
    plt.figure(figsize = (50, 50))
    plt.subplot(131)
    plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    

"""RUIDO SAL Y PIMIENTA EN LA IMAGEN A COLOR"""
def salypimienta_color(im,prob):
    tiempoIn = time.time()    
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    [ren, col] = im.size
    sal = Image.new("RGB",(ren, col))
    for i in range(ren):
        for j in range(col):
            r,g,b = im.getpixel((i, j))
            if random.random() < prob:
                syp = random.randint(0,1)
                if syp == 0:
                    syp = 0
                else:
                    syp = 255
                sal.putpixel((i, j),(syp, syp, syp))
            else:
                sal.putpixel((i, j),(r, g, b))
    sal.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""RUIDO SAL Y PIMIENTA EN LA IMAGEN A GRISES"""
def salypimienta_grises(im,prob):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    [ren, col] = im.size
    sal = im
    nMaxRen = round(ren * prob / 100.0)
    nMaxCol = round(col * prob / 100.0)
    i = 1
    for i in range(nMaxRen):
        j = 1
        for j in range(nMaxCol):
            cx = round(np.random.rand() * (col - 1)) + 1
            cy = round(np.random.rand() * (ren - 1)) + 1
            aaa = round(np.random.rand() * 255)
        if aaa > 128:
            val = 255
            sal.putpixel((cy, cx),(val))
        else:
            val= 1
            sal.putpixel((cy, cx),(val))
    sal.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""FILTRO MAXIMO DE LA IMAGEN A GRISES PARA QUITAR RUIDO"""
def filtro_maximo(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    out = im
    [ren, col] = out.size
    matriz = np.asarray(out, dtype = np.float32)
    i = 0
    while i < ren - 3:
        j = 0
        while j < col - 3:
            submatriz = matriz[j:j+3,i:i+3]
            submatriz = submatriz.reshape(1, 9)
            nuevo = int(max(max(submatriz)))
            out.putpixel((i + 1, j + 1),(nuevo))
            j+=1
        i+=1
    out.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""FILTRO MINIMO DE LA IMAGEN A GRISES PARA QUITAR RUIDO"""
def filtro_minimo(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    out = im
    [ren, col] = out.size
    matriz = np.asarray(out, dtype = np.float32)
    i = 0
    while i < ren - 3:
        j = 0
        while j < col - 3:
            submatriz = matriz[j:j+3,i:i+3]
            submatriz = submatriz.reshape(1, 9)
            nuevo = int(min(min(submatriz)))
            out.putpixel((i + 1, j + 1),(nuevo))
            j+=1
        i+=1
    out.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""FILTRO MEDIANA DE LA IMAGEN A GRISES PARA QUITAR RUIDO SAL Y PIMIENTA"""
def filtro_mediana(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    out = im
    [ren, col] = out.size
    matriz = np.asarray(out, dtype = np.float32)
    i = 0
    while i < ren - 3:
        j = 0
        while j < col - 3:
            submatriz = matriz[j:j+3,i:i+3]
            submatriz = submatriz.reshape(1, 9)
            nuevo = (max(submatriz))
            nuevo = statistics.median(nuevo)
            nuevo = int(nuevo)
            out.putpixel((i + 1, j + 1),(nuevo))
            j+=1
        i+=1
    out.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    

"""FILTRO MODA DE LA IMAGEN A GRISES PARA QUITAR RUIDO SAL Y PIMIENTA"""
def filtro_moda(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    out = im
    [ren, col] = out.size
    matriz = np.asarray(out, dtype = np.float32)
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
            out.putpixel((i + 1, j + 1),(nuevo))
            j+=1
        i+=1
    out.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
            
"""DETECCION DE BORDES CON SOBEL EN UNA IMAGEN A GRISES"""
def bordes_sobel(im, mask):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    [ren, col] = ima.size
    pix = ima.load()
    out_im = Image.new("L", (ren, col))
#   gx + gy + prewit45° = ([1,3,3],[-3,-2,3],[-3,-3,1])
#   gx = ([-1,0,1], [-2,0,2], [-1,0,1])
#   gy = ([1,2,1], [0,0,0], [-1,-2,-1])   
    out = out_im.load()
    for i in range(ren):
        for j in range(col):
            suma = 0
            for n in range(i-1, i+2):
                for m in range(j-1, j+2):
                    if n >= 0 and m >= 0 and n < ren and m < col:
                        suma += mask[n - (i - 1)][ m - (j - 1)] * pix[n, m]
            out[i, j] = suma
    out_im.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""DETECCION DE BORDES CON CANNY EN UNA IMAGEN A GRISES"""
def bordes_canny(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    ima = ndi.gaussian_filter(im, 4)
    edges = feature.canny(ima)
    fig, (ax2) = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 3), sharex = True, sharey = True)
    ax2.imshow(edges, cmap = plt.cm.gray)
    ax2.axis('off')
    plt.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')


"""UMBRAL OPTIMO POR EL METODO OTSU Y APLICACION DE UMBRALIZACION AL "ARRAY" EN UNA IMAGEN A GRISES"""
def umbral_otsu(im):
    tiempoIn = time.time()
    ruta = ("C:/Users/CkriZz/Pictures/" + im)
    im = Image.open(ruta)
    im.show()
    ima = im
    width, height = ima.size
    img = np.array(ima.getdata())
    histogram = np.array(ima.histogram(),float) / (width * height)
    #Vector de probabilidad acomulada.
    omega = np.zeros(256)
    #Vector de media acomulada
    mean = np.zeros(256)
    #Partiendo del histograma normalizado se calculan la probabilidad
    #acomulada (omega) y la media acomulada (mean)
    omega[0] = histogram[0]
    for i in range(len(histogram)):
        omega[i] = omega[i - 1] + histogram[i]
        mean[i] = mean[i - 1] + (i - 1) * histogram[i]
    sigmaB2 = 0
    mt = mean[len(histogram) - 1] #El Valor de la intensidad media de la imagen
    sigmaB2max = 0
    T = 0
    for i in range(len(histogram)):
        clase1 = omega[i]
        clase2 = 1 - clase1
        if clase1 != 0 and clase2 != 0:
            m1 = mean[i] / clase1
            m2 = (mt - mean[i]) / clase2
            sigmaB2 = (clase1 * (m1 - mt) * (m1 - mt) + clase2 * (m2 - mt) * (m2 - mt))
            if sigmaB2 > sigmaB2max:
                sigmaB2max = sigmaB2
                T = i
    thr = int(T)
    print('El Umbral Optimo De La Imagen Es: ' ,thr)
    #Se Aplica la umbralización al "array" de la imagen
    #limites de procesado en x
    x_min, x_max = 0, width
    #limites de procesado en y
    y_min, y_max = 0, height
    #imagen de salida
    img_out = np.zeros(width * height)
    #procesado de la imagen
    loc = 0 #posicin del "pixel" actual
    for y in range (y_min, y_max):
        for x in range(x_min, x_max):
            loc = y * width + x
            if img[loc] > thr:
                img_out[loc] = 255
            else:
                img_out[loc] = 0
    img_thr = img_out
    im_otsu = img_thr.reshape(height, width)
    im_otsu = Image.fromarray(im_otsu)
    im_otsu.show()
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
    
"""MATRIZ DE CONCURRENCIA"""
def matrizConcurrencia(datos):
#   datos = ([0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3])
    tiempoIn = time.time()
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
    print("-----45º Grados-----")
    """45º Grados"""
    cont = 1
    i = 1
    cuarenta = np.zeros((x + 1, x + 1))
    while i < (total - (ren)) + 1:
        n1 = nm[i]
        n2 = nm[i + 3]
        cuarenta[n1, n2] = cuarenta[n1, n2] + 1
        cuarenta[n2, n1] = cuarenta[n2, n1] + 1
        if(cont == (col-1)):
            i = i + 2
            cont = 1
        else:
            i = i + 1
            cont = cont + 1
    print(cuarenta)    
    print("-----90º Grados-----")
    """90º Grados"""
    cont = 1
    i = 0
    noventa = np.zeros((x + 1, x + 1))
    while i < (total - (ren)):
        n1 = nm[i]
        n2 = nm[i + 4]
        noventa[n1, n2] = noventa[n1, n2] + 1
        noventa[n2, n1] = noventa[n2, n1] + 1
        i = i + 1
    print(noventa)    
    print("-----135º Grados-----")
    """135º Grados"""
    cont = 1
    i = 1
    cien = np.zeros((x + 1, x + 1))
    while i < (total - (ren)) - 1:
            n1 = nm[i];
            n2 = nm[i + 5];
            cien[n1, n2] = cien[n1, n2] + 1;
            cien[n2, n1] = cien[n2, n1] + 1;
            if(cont == (col - 1)):
                i = i + 2
                cont = 1
            else:
                i = i + 1
                cont = cont + 1
    print(cien)    
    print("--------------------")
    tiempoFin = time.time()
    print('El Proceso Tardo: ', tiempoFin - tiempoIn, ' Segundos')
    
