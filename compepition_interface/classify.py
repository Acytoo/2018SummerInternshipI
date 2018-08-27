from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.externals import joblib

def trainmodel(inData):
    print("[train] start")
    pca=PCA(256)

    feats,types=inData
    pca.fit(feats)
    if feats.shape[1] >256:
        feats=pca.transform(feats)

    normalizer=preprocessing.Normalizer().fit(feats)
    feats=normalizer.transform(feats)

    svm=SVC(C=10000,gamma=0.0001,kernel='linear',decision_function_shape='ovr',probability=True)
    #svm = MLPClassifier()
    #print(feats.shape)
    #print(types.shape)
    svm.fit(feats,types)

    joblib.dump(svm,"svm.pkl")
    joblib.dump(pca,"pca.pkl")
    joblib.dump(normalizer,"normalizer.pkl")

    print("[train] end")