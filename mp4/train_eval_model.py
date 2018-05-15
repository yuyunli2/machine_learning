"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers

from models.linear_model import LinearModel
from models.support_vector_machine import SupportVectorMachine
import random

def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    dataNum = len(data['image'])

    number = int(dataNum / batch_size)
    if (number * batch_size != dataNum):
        number = number + 1

    dataset = []
    for i in range(0, dataNum):
        row = []
        row.extend(data['image'][i])
        row.extend(data['label'][i])
        dataset.append(row)

    dataset = np.array(dataset)

    while num_steps > 0:
        num_steps = num_steps - 1
        # print(num_steps)
        if shuffle is True:
            np.random.shuffle(dataset)

        for i in range(1, number):
            x_value = dataset[batch_size * (i - 1):batch_size * i, :-1]
            y_value = dataset[batch_size * (i - 1):batch_size * i, -1].reshape(x_value.shape[0], 1)
            update_step(x_value, y_value, model, learning_rate)

        x2_value = dataset[batch_size * (number - 1):, :-1]
        y2_value = dataset[batch_size * (number - 1):, -1].reshape(x2_value.shape[0], 1)

        update_step(x2_value, y2_value, model, learning_rate)

    print('w', model.w)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)

    grad = learning_rate * model.backward(f, y_batch)

    model.w = model.w - grad


def train_model_qp(data, model):
    """Computes and sets the optimal model weights (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    pass
    print('z', z)
    # Set model.w
    model.w = z[:data['image'].shape[1]+1]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    y = data['label']
    x = data['image']
    newList = np.ones((x.shape[0], 1))
    x = np.append(x, newList, axis=1)

    N = x.shape[0]
    dim = model.ndims+1
    P = np.zeros((dim + N, dim + N))
    for i in range(dim):
        P[i][i] = model.w_decay_factor

    q = np.zeros((dim + N, 1))
    for i in range(dim, dim + N):
        q[i] = 1

    G1 = -y * x
    print("g1", G1.shape)
    G2 = -np.eye(N)
    G3 = np.zeros((N,dim))
    G4 = -np.eye(N)
    G12 = np.append(G1, G2, 1)
    # print('G12',G12.shape)
    G34 = np.append(G3, G4, 1)
    # print('G34', G34.shape)
    G = np.append(G12, G34, 0)

    h1 = -np.ones((N,1))
    h2 = np.zeros((N,1))
    h = np.append(h1, h2, 0)
    # print(P,q,G,h)
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    # f = np.matmul(model.x, model.w)
    x = data['image']
    f = model.forward(x)
    loss = model.total_loss(f, data['label'])
    acc = 0

    y_predict = model.predict(f)
    # print(y_predict)
    accuracy = y_predict == data['label']
    # print('accuracy', accuracy)
    # print(y_predict)
    # print(data['label'])
    acc = sum(accuracy) / len(accuracy)
    return loss, acc
