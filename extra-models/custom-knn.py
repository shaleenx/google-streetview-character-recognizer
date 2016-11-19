#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Authors: Shaleen Kumar Gupta | Shruti Singh | November 2016

import time
import numpy as np
import csv
from sklearn.metrics import accuracy_score as score
from skimage.io import imread
from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.grid_search import GridSearchCV
import pickle

def euclidean_distance(a, b):
    dif = a - b
    return dif.dot(dif)

def get_k_nearest_neighbors(x, i, k):
    imageI = x[i, :]
    distances = [euclidean_distance(imageI, x[j, :]) for j in range(x.shape[0])]
    sortedNeighbors = np.argsort(distances)
    kNearestNeighbors = sortedNeighbors[1:(k+1)]
    return kNearestNeighbors

def assign_label(x, y, k, i):
    kNearestNeighbors = get_k_nearest_neighbors(x, i, k)
    counts = {}
    highestCount = 0
    mostPopularLabel = 0
    for n in kNearestNeighbors:
        labelOfN = y[n]
        if labelOfN not in counts:
            counts[labelOfN] = 1
        counts[labelOfN] += 1
        if counts[labelOfN] > highestCount:
            highestCount = counts[labelOfN]
            mostPopularLabel = labelOfN
    return mostPopularLabel

def read_images(path, labelsInfo, typeData, imageSize):
    x = np.zeros((labelsInfo.shape[0], imageSize))

    for (index, idImage) in enumerate(labelsInfo[:, 0]):
        fileName = "{0}/data/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        image = imread(fileName, as_grey=True)

        x[index, :] = np.reshape(image, (1, imageSize))

    return x

imageSize = 400

path = "../"

with open('{0}/data/trainLabels.csv'.format(path)) as csvfile:
    labelsInfoTrain = np.array(list(csv.reader(csvfile, delimiter=',')))

labelsInfoTrain = labelsInfoTrain[1:, :]

start = time.time()
xTrain = read_images(path, labelsInfoTrain, "train", imageSize)
print(time.time() - start , "seconds elapsed in reading training data")

with open('{0}/data/sampleSubmission.csv'.format(path)) as csvfile:
    labelsInfoTest = np.array(list(csv.reader(csvfile, delimiter=',')))

labelsInfoTest = labelsInfoTest[1:, :]

start = time.time()
xTest = read_images(path, labelsInfoTest, "test", imageSize)
print(time.time() - start , "seconds elapsed in reading testing data")

yTrain = np.array(list(map(ord, labelsInfoTrain[:, 1])))

start = time.time()

k = 1

yPredictions = [assign_label(xTrain, yTrain, k, i) for i in range(xTrain.shape[0])]

print("Training Time:", time.time()-start, "seconds")

print("Accuracy Score", score(yTrain, yPredictions))

print("Dummy Prediction", chr(yPredictions[27]))

#model = knn(n_neighbors = 1)

#model.fit(xTrain, yTrain)

#print(model.score(xTrain, yTrain))

#print(model.predict(xTrain[2]))

# save the model to disk
#filename = 'model.sav'
#pickle.dump(model, open(filename, 'wb'))

# adjust the number of folds later
#crossvalAccuracy = (k_fold_CV(model, xTrain, yTrain, cv=2, scoring="accuracy"))
#crossvalAccuracy = np.mean(crossvalAccuracy)
#
#print("The 2-fold cross validation accuracy of 1NN:", crossvalAccuracy)
#print(time.time() - start, "seconds elapsed in running 2 fold CV on 1NN")
#
## Tuning the value of k
#
#start = time.time()
#tunedParameters = [{"n_neighbors":list(range(1,5))}]
#classifier = GridSearchCV(model, tunedParameters, cv=5, scoring="accuracy")
#classifier.fit(xTrain, yTrain)
#
#print(classifier.grid_scores_)
#print(time.time() - start , "seconds elapsed in fitting classifier")
