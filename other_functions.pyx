'''
custom functions
'''
from cv2 import cv2
import os
import os
import shutil
from multiprocessing import Pool
import numpy as np

def folder_checker(path):
    if not(os.path.isdir(path)):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def __data_loader(arg):
    path = arg[0]
    shape = arg[1]
    img = cv2.resize(
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), shape, cv2.INTER_LANCZOS4)
    return img

def batch_img_reader(path, shape=(720, 480), datatype="png"):
    def format_checker(name, datatype):
        return (datanamelist[0][-len(datatype)::] == datatype)
    
    data = []
    datanamelist = os.listdir(path)
    arg = [[path+"\\"+i, shape] for i in datanamelist if(format_checker(i, ".png"))]

    pool = Pool()
    data = pool.map(__data_loader, arg)
    pool.close()
    pool.join()
    print("Load " + path + " done.")
    return data
