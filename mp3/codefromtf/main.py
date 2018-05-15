"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.01
max_iters = 100

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    # Initialize model.
    model = LogisticModel_TF(A.shape[1]-1,'gaussian')
    # Build TensorFlow training graph
    model.build_graph(learn_rate)
    # Train model via gradient descent.
    model.fit(T, A, max_iters)
    # Compute classification accuracy based on the return of the "fit" method



    
if __name__ == '__main__':
    tf.app.run()
