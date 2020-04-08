import numpy as np
import sys
import os
import shutil
import math
from matplotlib import pyplot as plt

class Raindrop(object):

    def __init__(self):
        self.video = object

    def RainDropRemoval(self, YUVpath, frames, rows, cols, referFrames=50):
        try:
            start = frames[0]
            end = frames[1]
        except:
            start = 0
            end = frames

        video = self.Reader(YUVpath, frames, rows, cols)

        print("Rain drop removal...", end="")
        Y = np.array([i[0] for i in video])  # (frame, rows, cols)
        newY = np.array(Y)
        referMaps = [Y[i] for i in range(len(Y)) if(i < referFrames)]

        meanMap, leftMeanMap = self.__ReferFramesStatistics(referMaps, rows, cols)
            
        oldestMap = np.array([])
        for frame in range(end):
            print("\rRain drop removal...", frame, end="")
            for x in range(rows):
                for y in range(cols):
                    if(Y[frame, x, y] > meanMap[x, y]):
                        try:
                            if(leftMeanMap[x, y, 1] > 1):
                                # print(leftMeanMap[x, y], meanMap[x, y], oldestMap[x, y], Y[frame, x, y])
                                newY[frame, x, y] = int(leftMeanMap[x, y, 0])
                            else:
                                newY[frame, x, y] = int(meanMap[x, y])
                        except:
                            # print(leftMeanMap[x, y], meanMap[x, y], oldestMap[x, y], Y[frame, x, y])
                            newY[frame, x, y] = int(meanMap[x, y])
            
            # 更新(待改進)
            oldestMap = referMaps.pop(0)
            referMaps.append(Y[frame])
            meanMap, leftMeanMap = self.__Refresh(referFrames, meanMap, leftMeanMap, oldestMap, Y[frame], rows, cols)
            # meanMap, leftMeanMap = self.__ReferFramesStatistics(referMaps, rows, cols)

        # 寫回
        for frame in range(end):
            video[frame] = [newY[frame], video[frame][1], video[frame][2]]

        self.video = video
        print("\rRain drop removal...OK.             ")

    def Exporter(self, outputPath, frames, rows, cols):
        print("Export...")
        fp = open(outputPath, 'wb')

        uv_rows = rows//2
        uv_cols = cols//2

        for frame in range(frames):
            for row in range(rows):
                for col in range(cols):
                    fp.write(np.ubyte(self.video[frame][0][row, col]))
            for uv in range(1, 3):
                for row in range(uv_rows):
                    for col in range(uv_cols):
                        fp.write(np.ubyte(self.video[frame][uv][row, col]))
        fp.close()
        print("OK.")

    def Reader(self, YUVpath, frames, rows, cols):
        print("Reading...", end="")
        fp = open(YUVpath, 'rb')

        uv_rows = rows//2
        uv_cols = cols//2

        try:
            start = frames[0]
            end = frames[1]
        except:
            start = 0
            end = frames

        allFrames = []
        for frame in range(end):
            print("\rReading...", frame, end="")
            if(frame < start):
                continue
            else:
                Y = np.zeros((rows, cols))
                U = np.zeros((uv_rows, uv_cols))
                V = np.zeros((uv_rows, uv_cols))
                for m in range(rows):
                    for n in range(cols):
                        Y[m, n] = ord(fp.read(1))
                for m in range(uv_rows):
                    for n in range(uv_cols):
                        V[m, n] = ord(fp.read(1))
                for m in range(uv_rows):
                    for n in range(uv_cols):
                        U[m, n] = ord(fp.read(1))

                allFrames.append([Y, U, V])

        fp.close()
        print("\rReading...OK.                    ")
        return allFrames  # (frame, [Y,U,V], rows, cols)

    def __Refresh(self, referFrames, meanMap, leftMeanMap, oldestMap, newestMap, rows, cols):
        leftMeanMap_out = np.array(leftMeanMap)
        meanMap_out = np.array(meanMap)
        
        for row in range(rows):
            for col in range(cols):
                popInAndOut = (newestMap[row, col] - oldestMap[row, col]) / referFrames

                if(oldestMap[row, col] > meanMap[row, col]):
                    if(newestMap[row, col] > meanMap[row, col]):
                        pass
                    else:
                        if((not np.isnan(leftMeanMap[row, col, 0])) and (not np.isinf(leftMeanMap[row, col, 0]))):
                            leftMeanMap_out[row, col, 1] += 1
                            leftMeanMap_out[row, col, 0] = (leftMeanMap[row, col, 0]*leftMeanMap[row, col, 1]+newestMap[row, col]) / leftMeanMap_out[row, col, 1]
                        else:
                            leftMeanMap_out[row, col, 1] += 1
                            leftMeanMap_out[row, col, 0] = (newestMap[row, col]) / leftMeanMap_out[row, col, 1]
                else:
                    if(newestMap[row, col] > meanMap[row, col]):
                        if((not np.isnan(leftMeanMap[row, col, 0])) and (not np.isinf(leftMeanMap[row, col, 0]))):
                            if(leftMeanMap[row, col, 1] > 1):
                                leftMeanMap_out[row, col, 1] -= 1
                                leftMeanMap_out[row, col, 0] = (leftMeanMap[row, col, 0]*leftMeanMap[row, col, 1]-oldestMap[row, col]) / leftMeanMap_out[row, col, 1]
                            else:
                                leftMeanMap_out[row, col, 1] = 0
                                leftMeanMap_out[row, col, 0] = np.nan
                        else:
                            leftMeanMap_out[row, col, 1] = 0
                            leftMeanMap_out[row, col, 0] = np.nan
                    else:
                        if((not np.isnan(leftMeanMap[row, col, 0])) and (not np.isinf(leftMeanMap[row, col, 0]))):
                            leftMeanMap_out[row, col, 0] = (leftMeanMap[row, col, 0]*leftMeanMap[row, col, 1]+newestMap[row, col]-oldestMap[row, col]) / leftMeanMap_out[row, col, 1]
                        else:
                            leftMeanMap_out[row, col, 0] = np.nan

                meanMap_out[row, col] = meanMap[row, col] + popInAndOut

        return meanMap_out, leftMeanMap_out

    def __ReferFramesStatistics(self, referMaps, rows, cols):
        meanMap = np.zeros((rows, cols))
        leftMeanMap = np.zeros((rows, cols, 2))

        for row in range(rows):
            for col in range(cols):
                temp = np.array([frame[row, col] for frame in referMaps])
                meanMap[row, col] = temp.mean()
                leftTemp = np.array([frame[row, col] for frame in referMaps if(frame[row, col] <= meanMap[row, col])])
                leftMeanMap[row, col, 0] = leftTemp.mean()
                leftMeanMap[row, col, 1] = len(leftTemp)
                
        return meanMap, leftMeanMap

    def __FolderChecker(self, path):
        if not(os.path.isdir(path)):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)


if __name__ == "__main__":
    inputPath = "demo.yuv"
    frames = 150 #753
    rows = 288
    cols = 352

    raindrop = Raindrop()
    raindrop.RainDropRemoval(inputPath, frames, rows, cols, referFrames=50)
    raindrop.Exporter("./out/out.yuv", frames, rows, cols)

    # raindrop.video = raindrop.Reader(inputPath, 100, rows, cols)
    # raindrop.Exporter("./out/out.yuv", 100, rows,cols)
