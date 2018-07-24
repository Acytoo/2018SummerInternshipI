import os
import cv2
import random
import numpy as np
import shutil

'''
目录结构组织为(divide前):
.../
    rootDir/
            originDir/
                    bear/
                    bicycle/
                    bird/
                    ...

目录结构组织为(divide之后)：
.../
    rootDir/
            originDir/
                    bear/
                    bicycle/
                    bird/
                    ...
            train/
            test/
            originCSV.csv
            trainCSV.csv
            testCSV.csv
'''
class Graph:

    def __init__(self,rootDir):
        self.rootDir=rootDir
        self.originCsvName="originCSV.csv"
        self.trainCsvName="trainCSV.csv"
        self.testCsvName="testCSV.csv"

    def _writeToCSV(self,l,path):

        file=open(path,"w+")
        for i in l:
            file.write(i[0]+','+i[1]+"\n")

        file.close()

    def _readCsv(self,path):
        csv = open(path, "r")
        lines = csv.readlines()
        reList = []
        for i in lines:
            i = i.strip('\n')
            (path, type) = i.split(',')
            reList.append([path, type])

        csv.close()
        return reList

    def _copyAndGetList(self,l,dir):
        trainImgList = []
        for i in l:
            fileName = os.path.split(i[0])[1]
            newPath = os.path.join(dir, fileName)
            trainImgList.append([newPath, i[1]])

            shutil.copy(i[0],newPath)

        return trainImgList

    def isDivided(self):
        if os.path.exists(os.path.join(self.rootDir,self.trainCsvName)):
            return True

    def getGraph(self,path):
        [dirname,filename]=os.path.split(path)

        matrix=cv2.imread(path)

        return matrix

    def divideTrainTest(self,originName,trainSize=4/5):
        dirPath=os.path.join(self.rootDir,originName)
        if(not (os.path.isdir(dirPath) and os.path.exists(dirPath)) ):
            print("error dirPath")
            return

        rootDir=self.rootDir
        originCSV=os.path.join(rootDir,self.originCsvName)
        trainCSV=os.path.join(rootDir,self.trainCsvName)
        testCSV=os.path.join(rootDir,self.testCsvName)

        trainDirPath=os.path.join(rootDir,r"train")
        testDirPath=os.path.join(rootDir,r"test")


        if os.path.exists(trainDirPath):
            shutil.rmtree(trainDirPath)
        if os.path.exists(testDirPath):
            shutil.rmtree(testDirPath)
        os.mkdir(trainDirPath)
        os.mkdir(testDirPath)

        typeDirs=os.listdir(dirPath)
        #print(fileDirs)

        imgPathList=[]

        for type in typeDirs:
            typeDir=os.path.join(dirPath,type)
            imgNames=os.listdir(typeDir)

            for imgName in imgNames:
                imgPathList.append([os.path.join(typeDir,imgName),type])

        self._writeToCSV(imgPathList,originCSV)


        random.shuffle(imgPathList)
        trainList=random.sample(imgPathList,int(len(imgPathList)*trainSize))
        testList=[i for i in imgPathList if i not in trainList]

        trainImgList=self._copyAndGetList(trainList,trainDirPath)
        testImgList=self._copyAndGetList(testList,testDirPath)

        self._writeToCSV(trainImgList,trainCSV)
        self._writeToCSV(testImgList,testCSV)

        return [trainImgList,testImgList]

    #返回训练集图像的路径与类型
    #list [[path,type] ...]
    def readTrainCSV(self):
        trainCsvPath=os.path.join(self.rootDir,self.trainCsvName)

        return self._readCsv(trainCsvPath)

    # 返回测试集图像的路径与类型
    # list [[path,type] ...]
    def readTestCsv(self):
        testCsvPath = os.path.join(self.rootDir, self.testCsvName)

        return self._readCsv(testCsvPath)

    # 返回源图像的路径与类型
    # list [[path,type] ...]
    def readOriginCsv(self):
        originCsvPath = os.path.join(self.rootDir, self.originCsvName)

        return self._readCsv(originCsvPath)

g=Graph(r"E:\ds2018")
print(g.readTrainCSV())