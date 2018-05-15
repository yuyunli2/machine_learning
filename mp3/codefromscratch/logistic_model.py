"""logistic model class for binary classification."""

import numpy as np

class LogisticModel(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term, 
            self.W = [Bias, W1, W2, W3, ...] 
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W = np.zeros(shape=(1, ndims + 1))
        elif W_init == 'ones':
            self.W = np.ones(shape=(1, ndims + 1))
        elif W_init == 'uniform':
            self.W = np.random.uniform(0, 1, ndims + 1).reshape(1, ndims + 1)
        elif W_init == 'gaussian':
            self.W = np.random.normal(0, 0.1, ndims + 1).reshape(1, ndims + 1)
        else:
            print ('Unknown W_init ', W_init) 
        
    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print ('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample 
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # print('X', X.shape)
        # print('W', self.W.shape)
        xw = np.matmul(X, self.W.T)
        score = 1/(1+np.exp(-xw)).reshape(xw.shape[0],)
        print('score', score.shape)
        return score

    def backward(self, Y_true, X):
        """ Backward operation for logistic models. 
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # print('X', X.shape)
        # print('W', self.W.shape)

        # Y_true = Y_true.reshape(1, Y_true.shape[0])
        # xw = np.matmul(self.W, X.T)
        # print('xw',xw.shape)
        # yxw =  np.dot(Y_true, xw.T)
        # print('yxw', yxw)
        # # print('yxw',yxw.shape
        # inBracket = 1 + np.exp(yxw)
        # # print('inBracket = ', inBracket.shape)
        # gradient= -np.dot(Y_true, X) / inBracket
        # # print('gradient', gradient.shape)
        # return gradient
        number = X.shape[0]
        gradient = 0
        for i in range(0,number):
            wx = np.dot(X[i,:],self.W.T)
            gradient = gradient - Y_true[i]*X[i,:]/(1+np.exp(Y_true[i]*wx))
        # print('gradient', gradient.shape)
        return gradient


    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        predict = []
        # probability = 1 / (1 + np.exp(-np.matmul(X,self.W.T)))
        score = self.forward(X)
        print('score',score)
        for i in score:
            if i > 0.5:
                predict.append(1)
            else:
                predict.append(-1)
        predict = np.array(predict)
        print('classify probability', predict)

        return predict

    
    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################

        for i in range(0, max_iters):
            self.update_step(Y_true, X, learn_rate)

    def update_step(self, Y_true, X, learn_rate):
        """Performs on single update step, (i.e. forward then backward).

        Args:
            x_batch(numpy.ndarray): input data of dimension (N, ndims).
            y_batch(numpy.ndarray): label data of dimension (N, 1).
            model(LinearModel): Initialized linear model.
        """
        grad = learn_rate * self.backward(Y_true, X)

        self.W = self.W - grad


# model = LogisticModel(2,'gaussian')
# a = np.array([[1,0,0],[1,0,0],[-1,0,0]])
# b = model.forward(a)
# c = model.classify(a)
# d = [1,1,-1]
# print(model.forward(a))
# print(model.classify(a))
# z = d==c
# print(sum(z))