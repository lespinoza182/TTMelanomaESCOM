"""
@author: luisespinoza, manuelarreola
"""
from PIL import Image
from matplotlib import pyplot as plt
from numpy import array
from collections import Counter
from future.utils import iteritems
import functools
import numpy as np
import statistics
import math

summaries = {0.0: [(1.0242365045390383, 0.6914471928283035), (126.40197794109865, 45.77687582606286), (0.060063831495954574, 0.030717794981104433), (4.370861743142268e-05, 5.62369806099574e-05), (0.8709797624924934, 0.060302668690457986), (1.2281335657348589, 0.3317781009726293)], 1.0: [(3.383225884227825, 1.1037016568866371), (146.38428279895595, 71.25032873333409), (0.18411297625078762, 0.06653200009893352), (0.00047308312778469717, 0.0003914743469593683), (0.6521226309797367, 0.11039263753002759), (1.9805899089122163, 0.7749195306578894)]}

def openImage(path):
    img = Image.open(path)
    return img

def imgGray(img):
    return img.convert('L')

def imgMedian(img):
    [ren, col] = img.size
    matriz = np.asarray(img, dtype = np.float32)
    i = 0
    while i < ren - 3:
        j = 0
        while j < col - 3:
            submatriz = matriz[j:j+3,i:i+3]
            submatriz = submatriz.reshape(1,9)
            nuevo = (max(submatriz))
            nuevo = statistics.median(nuevo)
            nuevo = int(nuevo)
            img.putpixel((i+1,j+1), (nuevo))
            j += 1
        i += 1
    return img

def imgMask(imOtsu, width, height):
    centroImagen = [int(width/2), int(height/2)]
    radio = min(centroImagen[0], centroImagen[1], width-centroImagen[0], height-centroImagen[1])
    Y, X = np.ogrid[:height, :width]
    distanciaCentro = np.sqrt((X - centroImagen[0])**2 + (Y - centroImagen[1])**2)
    mask = distanciaCentro <= radio
    imMask = imOtsu.copy()
    sh1, sh2 = np.shape(imMask)
    imMask = np.asarray(imMask, dtype = np.float32)
    maskedImg = imMask.copy()
    maskedImg[~mask] = 255
    imgthr2 = maskedImg
    imgSinBorde = imgthr2.reshape(height,width)
    imgSinBorde = Image.fromarray(imgSinBorde)
    return imgSinBorde

def imgOtsu(img):
    width, height = img.size
    imgAux = np.array(img.getdata())
    histogram = np.array(img.histogram(),float) / (width * height)
    omega = np.zeros(256)
    media = np.zeros(256)
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
    imOtsu = imgMask(imOtsu, width, height)
    return imOtsu

def imgMorphOpening(img):
    width, height = img.size
    a = np.asarray(img, dtype = np.float32)
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
    b = np.asarray(imgErosion, dtype = np.float32)
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
    return imgAperturaNeg

def imgSum(imgGray, imgOpen):
    img = imgGray
    i  = 0
    while i < imgOpen.size[0]:
        j = 0
        while j < imgOpen.size[1]:
            if imgOpen.getpixel((i,j)) == 0:
                img.putpixel((i,j),0)
            j+=1
        i+=1
    return img

def extract(img):
    [row, col]  = img.size
    width, height = img.size
    imgCo = np.asarray(img, dtype = np.int)
    histogram = np.array(img.histogram(),int) / (width * height)
    nivelesGris = len(Counter(histogram))
    datos = imgCo
    datos=np.asarray(datos)
    [ren, col] = datos.shape
    total = ren * col
    nm = datos.reshape((1, total))
    nm = max(nm)
    x = max(nm)
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
    [ren, col]  = cero.shape
    entropy  = energy = contrast = homogeneity = disimility = None
    prob_ac = 0
    normalizer = functools.reduce(lambda x,y: x + sum(y), cero, 0)
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
            entropy += -1.0 * prob * log_prob
            if energy is None:
                energy = prob ** 2
                continue
            energy += prob ** 2
            if contrast is None:
                contrast = ((m - n)**2) * prob
                continue
            contrast += ((m - n)**2) * prob
            if homogeneity is None:
                homogeneity = prob / ((1 + abs(m - n))*1.0)
                continue
            homogeneity += prob / ((1 + abs(m - n))*1.0)
            if disimility is None:
                disimility = prob * abs(m - n)
                continue
            disimility += prob * abs(m - n)
    if abs(entropy) < 0.0000001: entropy = 0.0
    vector = [entropy, contrast, homogeneity, energy, prob_ac, disimility]
    return vector

def processImage(img):
    gray = imgGray(img)
    median = imgMedian(gray)
    otsu = imgOtsu(median)
    open = imgMorphOpening(otsu)
    area = imgSum(gray, open)
    vector = extract(area)
    return vector

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean (numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in iteritems(summaries):
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in iteritems(probabilities):
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

path = ("/home/luisespinoza/Documentos/TT2/Desarrollo/Images/MelanomaTraining/IMD417.bmp")
image = openImage(path)
vector = processImage(image)
best = predict(summaries, vector)
print(best)
