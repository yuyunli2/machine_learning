"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
import random

from utils import io_tools
from utils import data_tools

def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    # processed_dataset = np.array(processed_dataset)
    # print(processed_dataset)
    print(len(processed_dataset))
    print(len(processed_dataset[0]))
    dataNum = len(processed_dataset[0])
    number = int(dataNum / batch_size)
    if (number * batch_size != dataNum):
        number = number + 1
    # Change dataset from list to array
    data = []
    for i in range(0,dataNum):
        row = []
        row.extend(processed_dataset[0][i])
        row.extend(processed_dataset[1][i])
        data.append(row)
    # print(type(data))
    data = np.array(data)
    if (shuffle == True):
        for i in range(1,number):
            random.shuffle(data[batch_size*(i-1):batch_size*i,:])
        random.shuffle(data[batch_size*(number-1):,:])


    while (num_steps > 1):
        num_steps = num_steps - 1
        update_step(data[:,:-1],data[:,-1],model,learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = LinearRegression.forward(model,x_batch)
    grad = learning_rate * LinearRegression.backward(model,f, y_batch)
    model.w = model.w - learning_rate * grad


dataset = io_tools.read_dataset('train.csv')
# print(dataset)
data = data_tools.preprocess_data(dataset)
ndim = data[0].shape[1]
print('data[0]',data[0])
print('ndim',ndim)
# print(data)
train_model(data,LinearRegression(ndim))

def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    # model.w = model.w -


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    loss = None

    return loss
