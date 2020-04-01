import numpy as np 
import matplotlib.pyplot as plt
from cv2 import cv2
from PIL import Image
import time
import random
import sys
import os
import shutil
import queue
import csv
import threading

class Raindrop(object):

    def __init__(self, savePath):
        self.savePath = savePath

    def Statistics(self, YUVpath, frames, rows, cols, seed=8006, points=50):
        def __hsitogramSaver(inputArray, fileName, arg=[]):
            histo, _ = np.histogram(inputArray)
            plt.figure()
            plt.hist(inputArray)
            try:
                _max = max(histo)
                for i in range(0, 6, 2):
                    plt.plot([arg[i],arg[i]],[0,_max], "k")

                plt.plot([arg[0]-arg[1],arg[0]-arg[1]], [0,_max], "r")
                plt.plot([arg[0]+arg[1],arg[0]+arg[1]], [0,_max], "r")
                plt.plot([arg[2]-arg[3],arg[2]-arg[3]], [0,_max], "g")
                plt.plot([arg[2]+arg[3],arg[2]+arg[3]], [0,_max], "g")
                plt.plot([arg[4]-arg[5],arg[4]-arg[5]], [0,_max], "b")
                plt.plot([arg[4]+arg[5],arg[4]+arg[5]], [0,_max], "b")
            except:
                pass
            plt.savefig(self.savePath+fileName)
            plt.close()
            time.sleep(0.01)
        
        def __mean_std(inputArray):
            meanAll = inputArray.mean()
            stdAll = inputArray.std()
            
            leftData = [i for i in inputArray if(i<=meanAll)]
            rightData = [i for i in inputArray if(i>meanAll)]
            leftData = np.array(leftData)
            rightData = np.array(rightData)

            meanLeft = leftData.mean()
            stdLeft = leftData.std()
            meanRight = rightData.mean()
            stdRight = rightData.std()
            return [meanAll, stdAll, meanLeft, stdLeft, meanRight, stdRight]
        # =========================================================================
        # random.seed(seed)
        # randomPoints = [(random.randint(0, rows), random.randint(0, cols)) for _ in range(points)]
        # randomPoints.sort(key=lambda x: x[0])
        randomPoints = []
        for row in range(rows):
            for col in range(cols):
                randomPoints.append((row, col))

        statisticsFrames = self.Reader(YUVpath, frames, rows, cols)
        Y = [i[0] for i in statisticsFrames]
        Y = np.array(Y) #(frames, rows, cols)

        self.__folderChecker(self.savePath)

        csvFile = open(self.savePath+"mean_std.csv", 'w', newline="")
        csvWriter = csv.writer(csvFile, delimiter=',')
        csvWriter.writerow(["global mean", "global std", "left hand mean", "left hand std", "right hand mean", "right hand std", "file name(x_y)"])
        threads = []
        for randomPoint in range(len(randomPoints)):
            eachPointY = [i[randomPoints[randomPoint]] for i in Y]
            eachPointY = np.array(eachPointY)

            # hist,bins = np.histogram(eachPointY, bins=histogramAxis)
            fileName = "("+str(randomPoints[randomPoint][0])+"_"+str(randomPoints[randomPoint][1])+")"+'.png'

            mean_stdData = __mean_std(eachPointY)

            __hsitogramSaver(eachPointY, fileName, mean_stdData)

            mean_stdData.append(fileName)
            csvWriter.writerow(mean_stdData)

        csvFile.close()

    def Reader(self, YUVpath, frames, rows, cols):
        fp = open(YUVpath,'rb')

        uv_rows = rows//2
        uv_cols = cols//2

        allFrames = []
        start = frames[0]
        end = frames[1]
        for frame in range(end):
            if(frame<start):
                continue
            else:
                Y = np.zeros((rows, cols))
                U = np.zeros((uv_rows, uv_cols))
                V = np.zeros((uv_rows, uv_cols))
                for m in range(rows):
                    for n in range(cols):
                        Y[m,n] = ord(fp.read(1))
                for m in range(uv_rows):
                    for n in range(uv_cols):
                        V[m,n] = ord(fp.read(1))
                for m in range(uv_rows):
                    for n in range(uv_cols):
                        U[m,n] = ord(fp.read(1))

                allFrames.append([Y, U, V])

        fp.close()
        return allFrames #(frame, 3, rows, cols)
    
    def __folderChecker(self, path):
        if not(os.path.isdir(path)):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

if __name__ == "__main__":
    filePath = "demo.yuv"
    frames = 753
    rows = 288
    cols = 352
    raindrop = Raindrop("./out/")
    raindrop.Statistics(filePath, [0, 50], rows, cols)

