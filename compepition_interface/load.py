import os
import cv2
from skimage import filters
import numpy as np


def loaddata(path):
    print("[load] start")
    typeDefine = ('bear', 'bicycle', 'bird', 'car', 'cow', 'elk', 'fox',
                  'giraffe', 'horse', 'koala', 'lion', 'monkey', 'plane',
                  'puppy', 'sheep', 'statue', 'tiger', 'tower', 'train',
                  'whale', 'zebra')
    types = os.listdir(path)
    data = []
    tags = []
    i = 0
    for type in types:
        typeDir = os.path.join(path, type)
        imgs = os.listdir(typeDir)
        for imgName in imgs:
            imgPath = os.path.join(typeDir, imgName)
            imgMat = cv2.imread(imgPath)
            if imgMat is None:
                continue

            #print("[load]:" + str(i))
            i += 1
            '''
            resolution: 100*100 is better than 96*96 and 128*128
            '''
            imgMat = cv2.resize(imgMat, (100, 100),
                                interpolation=cv2.INTER_CUBIC)

            b, g, r = cv2.split(imgMat)

            '''
            b, g, r is better than r, g, b
            '''
            train_data = np.vstack((b, g, r))

            data.append(train_data)
            tags.append(typeDefine.index(type))

    print("[load] end")

    return data, tags
