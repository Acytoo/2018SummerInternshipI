import numpy as np
from skimage import feature
from sklearn import preprocessing


class LBP:
    def __init__(self, p, r):
        self.p = p
        self.r = r

    def getVecLength(self):
        return 2**self.p

    def getFeature(self, imgMat):
        feat = feature.local_binary_pattern(
            imgMat, self.p, self.r, method='uniform')
        re, _ = np.histogram(feat, bins=range(
            256), normed=True)
        return re

    def getFeatVecs(self, imgList, load=0):
        if load == 1:
            feats = np.load(r"featVectLbp.npy")
            types = np.load(r"typesLbp.npy")
            return (feats, types)

        feats = None
        # i=0
        types = np.float32([]).reshape((0, 1))
        for mat, type in imgList:
            # print("[lbp]:"+str(i))
            # i+=1

            if mat is None:
                continue
            feat = self.getFeature(mat)

            if feats is None:
                feats = feat.reshape((1, -1))
            else:
                # print(feat.shape)
                # print(feats.shape)
                feats = np.append(feats, feat.reshape((1, -1)), axis=0)

            types = np.append(types, np.array(type).reshape((1, 1)))
        np.save(r"featVectLbp.npy", feats)
        np.save(r"typesLbp.npy", types)

        return (feats, types)


class HOG:

    def getVecLength(self):
        return 1764

    def getFeature(self, imgMat):
        feat = feature.hog(imgMat, orientations=9, pixels_per_cell=(
            16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
        feat = feat.reshape((1, -1))
        feat = preprocessing.normalize(feat)
        return feat

    def getFeatVecs(self, imgList, load=0):

        if load == 1:
            feats = np.load(r"featVectHog.npy")
            types = np.load(r"typesHog.npy")
            return (feats, types)

        feats = None
        # i=0
        types = np.float32([]).reshape((0, 1))
        for mat, type in imgList:
            # print("[hog]:"+str(i))
            # i+=1
            # print(mat.shape)
            feat = self.getFeature(mat)
            if feats is None:
                feats = feat.copy()
            else:
                feats = np.append(feats, feat, axis=0)

            types = np.append(types, np.float32([type]).reshape((1, 1)))
        np.save(r"featVectHog.npy", feats)
        np.save(r"typesHog.npy", types)
        return (feats, types)


def extractfeature(data, tags):
    print("[feature] start")
    matList = []

    for i in range(len(data)):
        matList.append((data[i], tags[i]))

    hog = HOG()
    lbp = LBP(8, 1)
    print("[feature]hog")
    featHog, types = hog.getFeatVecs(matList, load=0)
    print("[feature] lbp")
    featLbp, _ = lbp.getFeatVecs(matList, load=0)

    feats = np.append(featHog, featLbp, axis=1)
    # feats=featHog
    print("[feature] end")
    return (feats, types)
