#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import csv
from skimage.io import imread
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.grid_search import GridSearchCV

def read_image(path, labelsInfo, typeData, imageSize):
    x = np.zeros((labelsInfo.shape[0], imageSize))

    for (index, idImage) in enumerate(labelsInfo[1:, 0]):
        fileName = "{0}/data/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        image = imread(fileName, as_grey=True)

        x[index, :] = np.reshape(image, (1, imageSize))

    return x

imageSize = 400

path = "."

with open('{0}/data/trainLabels.csv'.format(path)) as csvfile:
    labelsInfoTrain = np.array(list(csv.reader(csvfile, delimiter=',')))

xTrain = read_image(path, labelsInfoTrain, "train", imageSize)

with open('{0}/data/sampleSubmission.csv'.format(path)) as csvfile:
    labelsInfoTest = np.array(list(csv.reader(csvfile, delimiter=',')))

xTest = read_image(path, labelsInfoTest, "test", imageSize)

print(labelsInfoTrain.shape)
labelsInfoTrain = labelsInfoTrain[1:, :]
print(labelsInfoTrain.shape)
print(labelsInfoTrain[:, 1])
yTrain = np.array(map(ord, labelsInfoTrain[:, 1]))

start = time.time()
model = knn(n_neighbors = 1)
# adjust the number of folds later
crossvalAccuracy = (k_fold_CV(model, xTrain, yTrain, cv=2, scoring="accuracy"))
crossvalAccuracy = np.mean(crossvalAccuracy)

print("The 2-fold cross validation accuracy of 1NN:", crossvalAccuracy)
print(time.time() - start, "seconds elapsed")

# Tuning the value of k

start = time.time()
tunedParameters = [{"n_neighbors":list(range(1,5))}]
classifier = GridSearchCV(model, tunes_parameters, cv=5, scoring="accuracy")
classifier.fit(xTrain, yTrain)

print(classifier.grid_scores_)
print(time.time() - start , "seconds elapsed")
