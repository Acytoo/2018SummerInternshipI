import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import shutil
import csv
import sys
import cv2

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
                    list_types.append(os.path.join(
                        root, each_file).split('\\')[-2])
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
    dataframe = pd.DataFrame({'pic_path': list_paths, 'pic_type': list_types})
    dataframe.to_csv(csv_name,  index=False, sep=',')


"""
Those functions work good when there isn't much data, but will consume much ram when the data sets get bigger
The folling functions solve this problem
Alec Chen 2018/07/21
"""


def save_info_csv(parent_folder, csv_name='image_info.csv'):
    """
    Prepare data for training and testing, now the data sets are in one directory and in the format of .jpg
    Args:  parent_folder:  Parent_folder:	Root path
    2018/07/21
    """
    csv_file = open(csv_name, 'w')
    writer = csv.writer(csv_file)
    for root, dirs, files in os.walk(parent_folder):
        for each_file in files:
            if '.jpg' in each_file:
                try:
                    csv_file.write(os.path.join(root, each_file)+',' +
                                   os.path.join(root, each_file).split('\\')[-2]+'\n')
                except Exception as e:
                    print(e)
    csv_file.close()


def image_resize_gray(root_dir='./neu-dataset', width=96, height=96, gray=True):
    """
    Resize image size using opencv, default is 96*96, gray
    Args:		root_dir
                            width
                            height
                            gray
    Returns:	None
    """
    if gray:
        for root, dirs, files in os.walk(root_dir):
            total = len(files)
            i = 1
            for each_file in files:
                print(i, ' / ', total)
                i += 1
                # only jpg now
                if os.path.splitext(each_file)[1] == '.jpg':
                    try:
                        # or 0:gray 1:color
                        img = cv2.imread(os.path.join(
                            root, each_file), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (width, height))
                        os.remove(os.path.join(root, each_file))
                        cv2.imwrite(os.path.join(root, each_file), img)
                    except Exception as e:
                        print(e)

    else:
        for root, dirs, files in os.walk(root_dir):
            total = len(files)
            i = 1
            for each_file in files:
                print(i, ' / ', total)
                i += 1
                # only jpg now
                if os.path.splitext(each_file)[1] == '.jpg':
                    try:
                        # or 0:gray 1:color
                        img = cv2.imread(os.path.join(
                            root, each_file), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (width, height))
                        os.remove(os.path.join(root, each_file))
                        cv2.imwrite(os.path.join(root, each_file), img)
                    except Exception as e:
                        print(e)


def distribute_train_test(ori_dir='ds2018', train_dir='train_dataset', test_dir='test_dataset', test_radio=0.2, random_seed=0):
    """
    Distribute train and test file and store file path, content type to csv file

    Before:
    neu-dateset
                    tiger
                    ...

    After:
    train_dataset
                    tiger
                    ...
    test_dataset
                    tiger
                    ...
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

    x = []
    for root, dirs, files in os.walk(ori_dir):
        for each_dir in dirs:
            if not os.path.exists(os.path.join(train_dir, each_dir)):
                os.makedirs(os.path.join(train_dir, each_dir))
            if not os.path.exists(os.path.join(test_dir, each_dir)):
                os.makedirs(os.path.join(test_dir, each_dir))

        for each_file in files:
            # only jpg image now
            if os.path.splitext(each_file)[1] == '.jpg':
                x.append(os.path.join(root, each_file))
    print('finish X')

    y = x.copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_radio, random_state=random_seed)

    try:
        csv_train = open('train_data.csv', 'w')
        csv_test = open('test_data.csv', 'w')
        i = 0
        for train_file in x_train:
            print(i)
            i += 1
            # .\neu_dataset\tiger\tiger1.jpg
            file_parts = train_file.split('/')
            print(file_parts)
            train_file_path = train_dir + '/' + \
                file_parts[1] + '/' + file_parts[2]
            csv_train.write(train_file_path+','+file_parts[1]+'\n')
            shutil.move(train_file, train_file_path)

        for test_file in x_test:
            file_parts = test_file.split('/')  # .\neu_dataset\tiger\tiger1.jpg
            test_file_path = test_dir + '/' + \
                file_parts[1] + '/' + file_parts[2]
            csv_test.write(test_file_path+','+file_parts[1]+'\n')
            shutil.move(test_file, test_file_path)

    except Exception as e:
        print(e)

    csv_train.close()
    csv_test.close()


if __name__ == '__main__':
    # names, types = get_img_path_type('.')
    # print(len(names))
    # print(len(types))
    # store_to_csv(names, types)

    # print('start move function')

    # distribute_train_test('tiger', 'yttrain', 'yttest')
    # if len(sys.argv) < 2:
    # 	print("Please enter root directory of your dataset")
    # 	exit(-1)
    # if len(sys.argv) == 2:
    # 	save_info_csv(sys.argv[1])	'''parent folder'''
    # else:
    # 	save_info_csv(sys.argv[1], sys.argv[2])		'''parent folder, file name'''
    # print('1')

    # image_resize_gray()
    input('pause')
    distribute_train_test()
