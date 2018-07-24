from sklearn import svm
from Graph.Graph import Graph
from Features.LBP import LBP
from sklearn.decomposition import PCA
import numpy as np

g=Graph(r"E:\ds2018")
if not g.isDivided():
    g.divideTrainTest("ds2018")

trainList=g.readTrainCSV()
features=np.float32([]).reshape(0,128*128)
types=[]
for imgPath,type in trainList:

    lbp=LBP([8],[1],"default")
    imgMatrix=g.getGreyGraph(imgPath)
    if imgMatrix is None:
        continue
    lbpFeatures=lbp.getFeature(imgMatrix)
    for j in lbpFeatures:
        print(features.shape)
        print(j.shape)
        np.append(features,j.reshape(1,128*128),axis=0)
        types.append(type)

clf=svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')

clf.fit(features,types)
print("done")