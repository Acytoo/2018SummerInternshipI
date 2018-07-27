"""
提取几种特征，将这些特征按照相同的顺序合并为一个向量，送入svm进行训练

考虑到每类中有不同的‘子类’。。

有一个train csv文件，包含所有的训练集

降维是必须的，否则内存会爆炸

降维操作很麻烦，如果训练前特征点降维了，而应用（实际图片分类时，如何保证特征点相同

svm.svc
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，
趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

Alec Chen
2018年7月26日21点01分

"""
import cv2
import csv
from sklearn import svm
import numpy as np
from sklearn.externals import joblib


def read_path_type(csv_path='./train_data.csv'):
	file = open(csv_path, 'r')
	csv_reader = csv.reader(file)
	for item in csv_reader:
		if csv_reader.line_num == 1:
			continue
		yield item[0], item[1]



def train_svm(features_file='features.npy', types_file='types.npy'):
	features = np.load(features_file)
	types = np.load(types_file)
	model = svm.SVC(gamma=0.0001, C=15, probability=True)
	model.fit(features, types)
	while True:
		file_name = input('your file to predict')
		if file_name == 'q':
			break
		img = cv2.imread(file_name, 0)

		orb = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
		sift = cv2.xfeatures2d.SIFT_create()
		surf = cv2.xfeatures2d.SURF_create()

		_, dp_orb = orb.detectAndCompute(img, None)
		kp_sift, dp_sift = sift.detectAndCompute(img, None)
		kp_surf, dp_surf = surf.detectAndCompute(img, None)
		feature = [dp_orb, dp_sift, dp_surf]
		print(model.predict(feature))



def orb_feature_extract():

	orb = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
	total_features = []
	total_types = []

	for img_path, img_type in read_path_type():
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			kp_orb, dp_orb = orb.detectAndCompute(img, None)
			if dp_orb is not None:
				total_features.append(dp_orb[0])
				total_types.append(img_type)
		except Exception as e:
			print(e)

	np.save('features_orb.npy', total_features)
	np.save('types_orb.npy', total_types)


def orb_train():

	total_features = np.load('features_orb.npy')
	total_types = np.load('types_orb.npy')

	model = svm.SVC(gamma=0.00001, C=20, probability=True)
	model.fit(total_features, total_types)
	print('fit finish')
	
	joblib.dump(model, 'orb.pkl')


def orb_detect():

	model = joblib.load('orb.pkl')
	orb = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
	while True:
		file_name = input('your file to predict\n')
		if file_name == 'q':
			break
		img = cv2.imread(file_name, 0)
		img = cv2.resize(img, (96, 96))
		_, dp_orb = orb.detectAndCompute(img, None)
		print(dp_orb)
		print(len(dp_orb))
		print(len(dp_orb[0]))

		if dp_orb is not None:
			# labels_predicted = model.predict(dp_orb).sort(reverse = True)
			# probabilites_predicted = model.predict_proba(dp_orb).sort(reverse = True)
			labels_predicted = sorted( model.predict(dp_orb), reverse=True)
			probabilites_predicted = sorted( model.predict_proba(dp_orb)[0], reverse=True)
			# print(labels_predicted[:5])
			# print(probabilites_predicted[:5])
			print(model.predict(dp_orb), len(model.predict(dp_orb)))
			print(model.predict_proba(dp_orb)[0], len(model.predict_proba(dp_orb)))
		else:
			print('Not a support image!')

 
def orb_judge():
	"""
	正确率
	"""

	model = joblib.load('orb.pkl')
	orb = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
	right = 0
	wrong = 0 

	for img_path, img_type in read_path_type('./test_data.csv'):
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			kp_orb, dp_orb = orb.detectAndCompute(img, None)
			if dp_orb is not None:		# no feature extracted
				res = model.predict(dp_orb)
				if res[0]:
					#res = sorted(res, reverse = True)
					if img_type in res[:5]:
						right += 1
						print('right')
					else:
						wrong += 1
				else:
					wrong += 1 
			else:
				wrong += 1
		except Exception as e:
			print(e)
			#pass

	print(right, wrong)
	print()

	'''There is a build-in function to calcualte this'''

	total_features = []
	total_types = []

	for img_path, img_type in read_path_type():
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			kp_orb, dp_orb = orb.detectAndCompute(img, None)
			if dp_orb is not None:
				total_features.append(dp_orb[0])
				total_types.append(img_type)
		except Exception as e:
			# print(e)
			pass
	print(model.score(total_features, total_types))





				
		
 	


def sift_feature_extract():

	sift = cv2.xfeatures2d.SIFT_create()
	total_features = []
	total_types = []

	for img_path, img_type in read_path_type():
		print(img_path)
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			kp_sift, dp_sift = sift.detectAndCompute(img, None)
			dp_sift_flatten = dp_sift.flatten()
			if dp_sift_flatten is not None:
				total_features.append(dp_sift_flatten)
				total_types.append(img_type)
		except Exception as e:
			print(e)

	np.save('features_sift.npy', total_features)
	np.save('types_sift.npy', total_types)





def sift_train():

	total_features = np.load('features_sift.npy')
	total_types = np.load('types_sift.npy')

	model = svm.SVC(gamma=0.001, C=15)
	try:
		model.fit(total_features, total_types)
	except Exception as e:
		print(e)
		print(len(total_features))
		print(len(total_types))
		print('error goback finish')
		exit(-1)
	print('fit finish')

	joblib.dump(model, 'sift.pkl') 


def sift_detect():
	model = joblib.load('sift.pkl')
	sift = cv2.xfeatures2d.SIFT_create()
	while True:
		file_name = input('your file to predict\n')
		if file_name == 'q':
			break
		img = cv2.imread(file_name, 0)

		kp_sift, dp_sift = sift.detectAndCompute(img, None)
		dp_sift_flatten = dp_sift.flatten()
		if dp_sift_flatten is not None:
			print(model.predict(dp_sift_flatten))
		else:
			print('None !!!!!')

	


def surf_part():

	sift = cv2.xfeatures2d.SIFT_create()
	surf = cv2.xfeatures2d.SURF_create()
	total_features = []
	total_types = []

	for img_path, img_type in read_path_type():
		print(img_path)
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			kp_orb, dp_orb = orb.detectAndCompute(img, None)
			# kp_sift, dp_sift = sift.detectAndCompute(img, None)
			# kp_surf, dp_surf = surf.detectAndCompute(img, None)
			# total_features.append(dp_sift)
			# total_types.append(img_type)
			if dp_orb is not None:
				total_features.append(dp_orb[0])
				total_types.append(img_type)
		except Exception as e:
			print(e)

# 	print(total_features[0])

	model = svm.SVC(gamma=0.01, C=5)
	model.fit(total_features, total_types)
	print('fit finish')



	while True:
		file_name = input('your file to predict\n')
		if file_name == 'q':
			break
		img = cv2.imread(file_name, 0)


		_, dp_orb = orb.detectAndCompute(img, None)
		# kp_sift, dp_sift = sift.detectAndCompute(img, None)
		# kp_surf, dp_surf = surf.detectAndCompute(img, None)
		if dp_orb is not None:
			feature = dp_orb
			print(model.predict(feature))
		else:
			print('None !!!!!')

	np.save('features_orb.npy', total_features)
	np.save('types_orb.npy', total_types)
	joblib.dump(model, 'orb.pkl') 









def test():


	orb = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
	sift = cv2.xfeatures2d.SIFT_create()
	surf = cv2.xfeatures2d.SURF_create()
	# hog = cv2.HOGDescriptor()
	total_features = []
	total_types = []

	interator = read_path_type()
	img = cv2.imread(next(interator)[0], 0)
	cv2.imshow('first', img)
	cv2.waitKey(0)
	# kp_orb, dp_orb = orb.detectAndCompute(img, None)
	# dp_orb_kpca = kpca.fit_transform(dp_orb)
	# dp_orb_back = kpca.inverse_transform(dp_orb_kpca)

	kp_sift, dp_sift = sift.detectAndCompute(img, None)
	# dp_sift_kpca = kpca.fit_transform(dp_sift)
	# dp_sift_back = kpca.inverse_transform(dp_sift_kpca)


	# kp_surf, dp_surf = surf.detectAndCompute(img, None)

	# dp_hog = hog.compute(img)
	# dp_hog_kpca = kpca.fit_transform(dp_hog)
	# dp_hog_back = kpca.inverse_transform(dp_hog_kpca)

	# each_features = [dp_orb, dp_sift, dp_surf, dp_hog]
	if dp_sift is not None:
		dp_sift = dp_sift.flatten()
		print(len(dp_sift))
		print(dp_sift)
	else:
		print('ooh, None')
	# print(len(dp_sift_kpca[0]))
	# print(dp_sift_kpca[0])
	# print(len(dp_sift_back[0]))
	# print(dp_sift_back[0])

	# print(len(dp_orb[0]))
	# print(dp_orb[0])
	# print(len(dp_orb_kpca[0]))
	# print(dp_orb_kpca[0])

	# print(len(dp_orb_back[0]))

	# print(dp_orb_backdp_orb_back[0])

	# print(len(dp_sift))
	# print(len(dp_surf))
	# print(len(dp_hog))
	# print(len(each_features))
	# print(each_features)
	# print()


	img = cv2.imread(next(interator)[0], 0)
	cv2.imshow('second', img)
	cv2.waitKey(0)
	# kp_orb, dp_orb = orb.detectAndCompute(img, None)
	kp_sift, dp_sift = sift.detectAndCompute(img, None)
	# kp_surf, dp_surf = surf.detectAndCompute(img, None)
	# dp_hog = hog.compute(img)
	# each_features = [dp_orb, dp_sift, dp_surf, dp_hog]
	dp_sift = dp_sift.flatten()
	print(len(dp_sift))
	print(dp_sift)
	# print(len(dp_surf))
	# print(len(dp_hog))
	# print(len(each_features))
	# print(each_features)
	# print()




 

if __name__ == '__main__':
	# sift_part()

	# orb_feature_extract()
	# orb_train()
	orb_detect()
	# orb_judge()

	# input('pause')
	# sift_feature_extract()
	# sift_train()