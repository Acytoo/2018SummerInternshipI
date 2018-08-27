from load import loaddata
from feature import extractfeature
from sklearn.externals import joblib

def testmodel(path):
    print("[test] start")
    data,tags=loaddata(path)

    feats,types=extractfeature(data,tags)

    svm=joblib.load("svm.pkl")
    pca = joblib.load("pca.pkl")
    normalizer = joblib.load("normalizer.pkl")

    feats=pca.transform(feats)
    feats=normalizer.transform(feats)

    probability=svm.predict_proba(feats)
    args = probability.argsort(axis=1)
    re=[]
    for i in range(args.shape[0]):
        if int(types[i]) in args[i][-5:]:
            re.append("True")
        else:
            re.append("False")

    print("[test] end")
    return re