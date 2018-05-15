"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 1000

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.
    model = LogisticModel(A.shape[1]-1,'uniform')
    # Train model via gradient descent.
    model.fit(T, A, learn_rate, max_iters)
    # score = model.forward(A)
    predict = model.classify(A)
    # print('predict',predict)
    # Save trained model to 'trained_weights.np'
    predict.tofile('trained_weights.np')
    # Load trained model from 'trained_weights.np'
    # a = np.array([])
    data = np.fromfile('trained_weights.np')
    # print(printT)
    # print(predict)
    print(T)
    a = predict == T
    # print()
    print(sum(a)/len(a))
    # Try all other methods: forward, backward, classify, compute accuracy

    pass 

