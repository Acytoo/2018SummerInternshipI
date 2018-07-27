from extract import read_path_type
import cv2
from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import os
from sklearn.model_selection import GridSearchCV


class Orb_detector(object):
	"""docstring for Orb_detector"""
	def __init__(self):
		super(Orb_detector, self).__init__()
		self.detector = cv2.ORB_create(nfeatures=15000, scoreType=cv2.ORB_FAST_SCORE)
		self.svm_model = svm.SVC(gamma=0.0001, C=5, probability=True, kernel='linear')


	def feature_extract(self):

		total_features = []
		total_types = []

		for img_path, img_type in read_path_type():
			try:
				print(img_path)
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
				kp_orb, dp_orb = self.detector.detectAndCompute(img, None)
				if dp_orb is not None:
					# if the number of extracted point is two small, double it until it works					
					while len(dp_orb) < 30:
						dp_orb = np.vstack((dp_orb, dp_orb))

					total_features.append(dp_orb[:30].flatten())
					total_types.append(img_type)
			except Exception as e:
				print(e)

		np.save('features_orb.npy', total_features)
		np.save('types_orb.npy', total_types)
		print('orb features extracted and stored')


	def train(self):

		total_features = np.load('features_orb.npy').tolist()
		total_types = np.load('types_orb.npy').tolist()
		print(len(total_features))
		print(len(total_types))
	
		self.svm_model.fit(total_features, total_types)
		
		joblib.dump(self.svm_model, 'orb.pkl')
		print('fit finish and model stored')



	def detect(self):

		if not os.path.exists('orb.pkl'):
			print('Model file not exist, please train your model first')
			exit(-1)

		self.svm_model = joblib.load('orb.pkl')
		while True:
			file_name = input('your file to predict\n')
			if file_name == 'q':
				break
			dp_orb = None
			try:
				img = cv2.imread(file_name, 0)
				img = cv2.resize(img, (96, 96))
				_, dp_orb = self.detector.detectAndCompute(img, None)
			except Exception as e:
				print(e)

			if dp_orb is not None:

				# zip_file = zip(model.classes_, model.predict_proba([dp_orb[0]])[0])
				# for (x,y) in zip_file:
				# 	print((x,y))

				# print('prediction:', model.predict([dp_orb[0]]))
				while len(dp_orb) < 30:
						dp_orb = np.vstack((dp_orb, dp_orb))
				decision = self.svm_model.decision_function([dp_orb[:30].flatten()])
				decision_zip = sorted(zip(self.svm_model.classes_, decision[0]), key=lambda x: x[1], reverse=True)

				for x,y in decision_zip[:5]:
					print(x,y)
			else:
				print('Not a support image!')





	def judge(self):
		"""
		正确率
		"""

		self.svm_model = joblib.load('orb.pkl')
		right = 0
		wrong = 0 

		for img_path, img_type in read_path_type('./test_data.csv'):
			try:
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
				# 已经resize了
				kp_orb, dp_orb = self.detector.detectAndCompute(img, None)
				if dp_orb is not None:		# no feature extracted
					while len(dp_orb) < 30:
						dp_orb = np.vstack((dp_orb, dp_orb))
					decision = self.svm_model.decision_function([dp_orb[:30].flatten()])
					decision_zip = sorted(zip(self.svm_model.classes_, decision[0]), key=lambda x: x[1], reverse=True)
					for x, _ in decision_zip[:5]:
						if img_type == x:
							right+=1
							continue
					wrong += 1
				else:
					wrong += 1
			except Exception as e:
				print(e)

		print(right, wrong)
		print()

		'''There is a build-in function to calcualte this'''

		total_features = []
		total_types = []

		for img_path, img_type in read_path_type('./test_data.csv'):
			try:
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
				kp_orb, dp_orb = self.detector.detectAndCompute(img, None)
				if dp_orb is not None:
					while len(dp_orb) < 30:
						dp_orb = np.vstack((dp_orb, dp_orb))
					total_features.append(dp_orb[:30].flatten())
					total_types.append(img_type)
			except Exception as e:
				# print(e)
				pass
		print(self.svm_model.score(total_features, total_types))


	def find_para(self):
		total_features = np.load('features_orb.npy')
		total_types = np.load('types_orb.npy')
		tunned_parameters=[
			{
				'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
				'C':[1,3,5,7,9,10,100],
				'gamma':[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
			}
		]
		test = GridSearchCV(svm.SVC(probability=True, decision_function_shape='ovo'), tunned_parameters)
		test.fit(total_features, total_types)
		print(test.best_params_)
		self.svm_model = test.best_estimator_
		joblib.dump(self.svm_model, 'orb_best.pkl')






if __name__ == '__main__':

	orb = Orb_detector()
	# orb.feature_extract()
	# orb.train()
	# orb.judge()
	# orb.detect()
	orb.find_para()