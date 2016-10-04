#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.io import imread

def read_image(path, labelsInfo, typeData, imageSize):
    x = np.zeros((label

    for (index, idImage) in enumerate(labelsInfo['ID']):
        fileName = "{0}/data/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        image = imread(fileName, as_grey=True)

        x[index, :] = np.reshape(image, (1, imageSize))

    return x

imageSize = 400

path = "."

with open('{0}/data/trainLabels.csv'.format(path)) as csvfile:
    labelsInfoTrain = np.array(list(csv.reader(csvfile, delimiter=',')))

with open('{0}/data/sampleSubmission.csv'.format(path)) as csvfile:
    labelsInfoTest = np.array(list(csv.reader(csvfile, delimiter=',')))

xTest = read_data(path, labelsInfoTest, "test", imageSize)

yTrain = map(
