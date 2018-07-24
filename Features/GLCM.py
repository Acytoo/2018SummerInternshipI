import cv2
import numpy as np
from skimage import feature
from Graph.Graph import Graph

class GLCM:

    def __init__(self,distance,angle):
        self.distance=distance
        self.angle=angle


    '''
    return :list of features
    '''
    def getFeature(self,imgMat):
        re=feature.greycomatrix(imgMat,self.distance,self.angle)
        features=[]
        for i in range(len(self.distance)):
            for j in range(len(self.angle)):
                features.append(re[:,:,i,j])

        return features


g=Graph(r"E:\ds2018")
trainList=g.readTrainCSV()
glcm=GLCM([1],[0,np.pi/4,np.pi/2,np.pi*3/4])
features=[]
print(trainList)
for img in trainList:
    #print(img[0])
    mat=g.getGreyGraph(img[0])
    if mat is None:
        print("None")
        continue
    features.extend(glcm.getFeature(mat))

print(features)