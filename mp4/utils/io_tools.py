"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = None
    data['label'] = None

    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    indexFile = open(data_txt_file, 'r')
    index = indexFile.read().replace('\n', ',').split(',')
    indexFile.close()
    # print(index)

    data['label'] = []
    number = int(len(index)/2)
    for i in range(number):
        data['label'].append(int(index[2*i+1]))
    data['label'] = np.array(data['label']).reshape(number, 1)
    # print(data['label'].shape)
    data['image'] = []
    for i in range(number):
        image = io.imread(image_data_path+ index[2*i] + '.jpg')
        data['image'].append(image)

    data['image'] = np.array(data['image'])
    # print(data['image'])
    # print(data['image'].shape,type(data['image']))

    return data

# train_set = read_dataset("data/train.txt", "data/image_data/")
# print(train_set)