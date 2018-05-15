"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    indexFile = open(path_to_dataset_folder+'/'+index_filename,'r')
    index = indexFile.read().split()
    indexFile.close()
    # print(index)
    # print(type(index))
    T = []
    A = []
    # print(len(index))
    length = int(len(index)/2)
    for i in range(0, length):
        row = [1]


        label = index[2*i]
        T.append(int(label))
        dataFile = open(path_to_dataset_folder+'/'+index[2*i+1],'r')
        data = dataFile.read().split()
        dataFile.close()
        # print(data)
        for ele in data:
            # print(ele)
            # print(type(ele))
            # print('ele=', float(ele))
            # print(row)
            row.append(float(ele))
        A.append(row)
        dataFile.close()
    # print(T)
    T = np.array(T)
    A = np.array(A)

    # print('T=', T)
    # print('A=', A)

    return A, T


# a, t = read_dataset(path_to_dataset_folder='data/trainset', index_filename='indexing.txt')
# print(a)
