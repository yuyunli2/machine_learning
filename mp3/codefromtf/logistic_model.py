"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term, 
            Weight = [Bias, W1, W2, W3, ...] 
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
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            self.W0 = tf.zeros([self.ndims + 1, 1])
        elif W_init == 'ones':
            self.W0 = tf.ones([self.ndims + 1, 1])
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform([self.ndims + 1, 1])
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal([self.ndims + 1, 1], mean=0.0, stddev=0.1)
        else:
            print ('Unknown W_init ', W_init) 
        

    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        # Initializing the variables

        self.W = tf.Variable(self.W0)

        # tf Graph Input
        self.X1 = tf.placeholder(tf.float32, shape=(1, self.W0.shape[0]))
        self.Y1 = tf.placeholder(tf.float32, shape=(1, 1))
        self.e = tf.placeholder(tf.float32)

        # Create Model
        score = tf.sigmoid(tf.matmul(self.X1, self.W))

        # Minimize the squared errors
        # cost = tf.reduce_sum(tf.pow(score - self.Y1, 2))  # L2 loss
        cost = (tf.pow(score - self.Y1, 2))  # L2 loss

        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)  # Gradient descent

        self.accuracy = tf.reduce_sum(self.e)/ tf.cast(tf.size(self.e),tf.float32)


    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################

        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            for i in range(max_iters):
                for (x, y) in zip(X, Y_true):
                    sess.run(self.optimizer, feed_dict={self.X1: x.reshape(1,X.shape[1]), self.Y1: y.reshape(1,1)})

                a = sess.run(self.W)
                c = []

                X2 = X.astype('float32')
                d = tf.sigmoid(tf.matmul(X2, a))
                scores = sess.run(d)

                for i in scores:
                    if i > 0.5:
                        c.append(1)
                    else:
                        c.append(0)

                e = c==Y_true
                print(sess.run(self.accuracy, feed_dict={self.e: e}))
            # print('accuracy', accuracy)
            print('scores', scores, scores.shape)
            return scores

    