from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def k_means(C):
    # Write your code here!
    data_read = pd.read_csv("data/data/iris.data")
    data = np.array(data_read)
    # print('data size', data.shape)
    length = data.shape[0]
    C = np.array(C)
    k = C.shape[0]
    m = C.shape[1]
    C_new = C
    C_old = np.zeros((k,m))
    count = 0

    while (np.linalg.norm(C_new-C_old) > 0.001):
        count+=1
        # print('count', count)
        cluster = {}
        for i in range(length):
            point = data[i,:m]
            # print('point', point)
            distance = []

            for j in range(k):
                distance.append(sum((point-C_new[j,:])**2))


            num = np.argmin(distance)
            if num not in cluster:
                cluster[num] = [point]
            else:
                cluster[num].append(point)
        # print('cluster', cluster)

        C_old = deepcopy(C_new)
        for ele in cluster:
            # print('ele', ele)
            C_new[ele,:] = np.mean(cluster[ele], axis=0)

        # print('C_new', C_new)

    C_final = C_new

    return C_final








