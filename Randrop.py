import os
import sys

import numpy as np
# from keras import Input, Model
# from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
#                           MaxPooling2D)
from matplotlib import pyplot as plt

from otherfunctions import batch_img_reader, data_shape_normalize


class Raindrop(object):

    def __init__(self, train, test):
        self.xtrain = train[0]
        self.ytrain = train[1]
        
        self.xtest = test[0]
        self.ytest = test[1]

        self.model = object

if __name__ == "__main__":
    dataroot = "D:\\Github\\NewRainDrop\\dataset"
    shape = (720, 480)
    xtrain = data_shape_normalize(batch_img_reader(dataroot+"\\train\\data"), shape)
    ytrain = data_shape_normalize(batch_img_reader(dataroot+"\\train\\gt"), shape)
    
    xtest = data_shape_normalize(batch_img_reader(dataroot+"\\test_a\\data"), shape)
    ytest = data_shape_normalize(batch_img_reader(dataroot+"\\test_a\\gt"), shape)

    raindrop = Raindrop([xtrain,ytrain], [xtest,ytest])
    # print(np.shape(raindrop.xtrain))
