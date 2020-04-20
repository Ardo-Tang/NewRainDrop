import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          MaxPooling2D, UpSampling2D, concatenate)
from keras.optimizers import Nadam
from matplotlib import pyplot as plt

from NewRainDrop.other_functions import batch_img_reader, folder_checker


class raindrop_removal(object):

    def __init__(self, train, valid, test):
        self.xtrain = train[0]
        self.ytrain = train[1]

        self.xvalid = valid[0]
        self.yvalid = valid[1]
        
        self.xtest = test[0]
        self.ytest = test[1]

        self.inputshape = np.shape(self.xtrain)[1::]
        self.model = object
        self.history = object
    
    def testmodel(self, outputpath):
        folder_checker(outputpath)
        out = self.model.predict(self.xtest)
        count = 0
        for i in out:
            plt.figure()
            plt.imshow(i)
            plt.savefig(outputpath+str(count)+".png")
            plt.close()

    def train(self):
        self.model = self.__build_model()
        es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
        csvLog = CSVLogger(filename='log.log')
        reduceLR = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=5)
        callbacks = [es, csvLog, reduceLR]
        self.history = self.model.fit(
            x=self.xtrain, y=self.ytrain, 
            batch_size=1024, epochs=10000, verbose=1,
            callbacks=callbacks,
            validation_data=(self.xvalid, self.yvalid),
            shuffle=True)

    def __build_model(self):
        inputs = Input(self.inputshape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Nadam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        # model.summary()
        return model

if __name__ == "__main__":
    dataroot = "D:\\Github\\NewRainDrop\\dataset"
    normalizevalue = 255.0
    train = [
        np.array(batch_img_reader(dataroot+"\\train\\data"))/normalizevalue, 
        np.array(batch_img_reader(dataroot+"\\train\\gt"))/normalizevalue]
    
    valid = [
        np.array(batch_img_reader(dataroot+"\\test_a\\data"))/normalizevalue, 
        np.array(batch_img_reader(dataroot+"\\test_a\\gt"))/normalizevalue]
    
    test = [
        np.array(batch_img_reader(dataroot+"\\test_b\\data"))/normalizevalue, 
        np.array(batch_img_reader(dataroot+"\\test_b\\gt"))/normalizevalue]

    remover = raindrop_removal(train, valid, test)
    remover.train()
    remover.testmodel("./out/")
