import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import shutil

'''
Read pictures' name from disk, store their path and type to a csv file with format {path, type}

Randomly distribute files to train folder and test folder, then store the file path and types

Alec Chen
2018/07/13
'''


def get_img_path_type(parent_folder):
	"""
	Args:		Parent_folder:	Root path
	Returns:	Two lists with paths and correspond types
	"""
	list_paths = []
	list_types = []
	for root, dirs, files in os.walk(parent_folder):
		for each_file in files:
			if '.jpg' in each_file:
				try:
					list_types.append(os.path.join(root, each_file).split('\\')[-2])
					list_paths.append(os.path.join(root, each_file))
				except Exception as e:
					print(e)


	return list_paths, list_types


def store_to_csv(list_paths, list_types, csv_name='pic_info.csv'):
	"""
	Args:		list_paths:	A list that contain all the paths
				list_types:	A list that contain correspond types
				csv_name:	csv file name that store the information, default is pic_info.csv
	returns:	None
	"""
	dataframe = pd.DataFrame({'pic_path':list_paths, 'pic_type':list_types})
	dataframe.to_csv(csv_name,  index = False, sep = ',')



def distribute_train_test(ori_dir, train_dir='train', test_dir='test', test_radio=0.2, random_seed=0):
	"""
	Args:		ori_dir:		Original file directory
				train_dir:		Train file directory
				test_dir:		Test file directory
				test_radio:		The radio of test files
				random_seed:	Random seed
	Returns:	None
	"""
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	x = y = os.listdir(ori_dir)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_radio, random_state=random_seed)

	try:
		for train_file in x_train:
			shutil.move(ori_dir+'/' + train_file, train_dir+'/' + train_file)

		for test_file in x_test:
			shutil.move(ori_dir+'/' + test_file, test_dir+'/' + test_file)
	except Exception as e:
		print(e)


if __name__ == '__main__':
	names, types = get_img_path_type('.')
	print(len(names))
	print(len(types))
	store_to_csv(names, types)

	print('start move function')


	distribute_train_test('tiger', 'yttrain', 'yttest')
