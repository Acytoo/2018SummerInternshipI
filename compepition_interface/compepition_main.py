from load import loaddata
from feature import extractfeature
from classify import trainmodel
from predict import testmodel


data, tags = loaddata("/home/acytoo/train_dataset/")

features = extractfeature(data, tags)

trainmodel(features)

res = testmodel("/home/acytoo/test_dataset/")

print "precise: ", res.count("True")*1.0/len(res)
