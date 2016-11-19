#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Authors: Shaleen Kumar Gupta | Shruti Singh | November 2016

import time
import numpy as np
import csv
from skimage.io import imread
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pickle

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

model = NearestCentroid(metric='euclidean', shrink_threshold=0)

model.fit(xTrain, yTrain)

print("Training Time:", time.time() - start, "seconds")

print("Accuracy Score:", model.score(xTrain, yTrain))

print("Dummy Prediction:", chr(model.predict(xTrain[0])[0]))

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
