from skimage import feature
import numpy as np
import cv2
from Graph.Graph import Graph

class LBP:

    '''
    ps:list
    rs:list
    methods:list    ("default","ror","uniform","var")
    '''
    def __init__(self,ps,rs,method="default"):
        self.p=ps
        self.r=rs
        self.method=method

    def getFeature(self,imgMat):
        features=[]
        for i in self.p:
            for j in self.r:
                features.append(feature.local_binary_pattern(imgMat,i,j,method=self.method))

        return features



g=Graph(r"E:\ds2018")
trainList=g.readTrainCSV()
lbp=LBP([8],[1])
features=[]
print(trainList)
for img in trainList:
    #print(img[0])
    mat=g.getGreyGraph(img[0])
    if mat is None:
        print("None")
        continue
    features.extend(lbp.getFeature(mat))


