#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Shaleen Kumar Gupta | Shruti Singh | November, 2016

import pickle
from sklearn.neighbors import KNeighborsClassifier as knn
from skimage.io import imread
import numpy as np
import time

class Recognizer:

    def read_images(path, numImages, imageSize):
        x = np.zeros((numImages, imageSize))

        for idImage in range(numImages):
            fileName = "{0}/input/{1}.Bmp".format(path, idImage)
            image = imread(fileName, as_grey=True)

            x[idImage, :] = np.reshape(image, (1, imageSize))

        return x

    def __init__():
        model = pickle.load(open("model.sav", "rb"))

    def recognize(path=".", numImages=2, imageSize=400):
        x = read_images(path, numImages, imageSize)
        val = ''.join(map(chr, model.predict(x)))
        print("Answer:", val)
        return val
