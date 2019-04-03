import csv
import math
import random

def loadData(filename):
    lines = csv.reader(open(filename, 'r'))
    #lines = csv.DictReader(filename)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean (numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(instances):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
            #print('X: ',x,'Mean: ',mean,'Stdev: ',stdev, 'Probability: ',probabilities[classValue])
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print(probabilities)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    print(bestLabel)
    return bestLabel

#Entrenamiento
#filename = 'melanoma.data.csv' #filename = 'nomelanoma.data.csv'
#dataset = loadData(filename)
#summaries = summarizeByClass(dataset)
#print(summaries)

#Entrenamiento al 100%
summaries = {0.0: [(1.0242365045390383, 0.6914471928283035), (126.40197794109865, 45.77687582606286), (0.060063831495954574, 0.030717794981104433), (4.370861743142268e-05, 5.62369806099574e-05), (0.8709797624924934, 0.060302668690457986), (1.2281335657348589, 0.3317781009726293)], 1.0: [(3.383225884227825, 1.1037016568866371), (146.38428279895595, 71.25032873333409), (0.18411297625078762, 0.06653200009893352), (0.00047308312778469717, 0.0003914743469593683), (0.6521226309797367, 0.11039263753002759), (1.9805899089122163, 0.7749195306578894)]}
#Entrenamiento al 75% -> 24 imagenes (12 positivas y 12 negativas) de training y 6 (3 positivas y 3 negativas) de test
#summaries =
#Entrenamiento al 50% -> 14 imagenes (7 positivas y 7 negativas) de training y 14 (7 positivas y 7 negativas) de test
#summaries =
inputVector = [0.21579129270132888,77.16793379060177,0.028038839012511575,0.000003622241387659875,0.9351102545207188,0.7845632333767888]
prediction = predict(summaries,inputVector)
print('Prediction: ',  prediction)
