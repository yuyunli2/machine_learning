"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """

    def __init__(self, ndims=784, nlatent=10):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        def xavier_init(size):
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

            return tf.random_normal(shape=size, stddev=xavier_stddev)

        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # self.D_W1 = tf.Variable(xavier_init([ndims, 128]), name='D_W1')
        # self.D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
        #
        # self.D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
        # self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
        #
        # # Input noise
        # print("nlatent:",nlatent)
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])
        #
        # self.G_W1 = tf.Variable(xavier_init([nlatent, 128]), name='G_W1')
        # self.G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
        #
        # self.G_W2 = tf.Variable(xavier_init([128, ndims]), name='G_W2')
        # self.G_b2 = tf.Variable(tf.zeros(shape=[ndims]), name='G_b2')



        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        # Generator Network Variables
        # gen_vars = [self.G_W1, self.G_W2,
        #             self.G_b1, self.G_b2]
        #
        # # Discriminator Network Variables
        # disc_vars = [self.D_W1, self.D_W2,
        #              self.D_b1, self.D_b2]

        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'D_' in var.name]
        g_vars = [var for var in tvars if 'G_' in var.name]


        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)

        self.train_gen = optimizer_gen.minimize(self.g_loss, var_list=g_vars)
        self.train_disc = optimizer_disc.minimize(self.d_loss, var_list=d_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """

        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            if (reuse):
                scope.reuse_variables()

            # xavier_stddev1 = 1. / tf.sqrt(in_dim / 2.)
            D_W1 = tf.get_variable('D_W1', [self._ndims, 128], initializer=layers.xavier_initializer(),trainable=True)
            # D_W1 = tf.get_variable('D_W1', [self._ndims, 128], initializer=xavier_init([self._ndims, 128]), trainable=True)
            D_b1 = tf.get_variable('D_b1', [128], initializer=tf.constant_initializer(0), trainable=True)

            D_W2 = tf.get_variable('D_W2', [128, 1], initializer=layers.xavier_initializer(), trainable=True)
            # D_W2 = tf.get_variable('D_W2', [128, 1], initializer=xavier_init([128, 1]), trainable=True)
            D_b2 = tf.get_variable('D_b2', [1], initializer=tf.constant_initializer(0), trainable=True)

            # hidden_layer = tf.matmul(x, self.D_W1)
            # hidden_layer = tf.add(hidden_layer, self.D_b1)
            # hidden_layer = tf.nn.relu(hidden_layer)
            # out_layer = tf.matmul(hidden_layer, self.D_W2)
            # y = tf.add(out_layer, self.D_b2)
            hidden_layer = tf.matmul(x, D_W1)
            hidden_layer = tf.add(hidden_layer, D_b1)
            hidden_layer = tf.nn.relu(hidden_layer)
            out_layer = tf.matmul(hidden_layer, D_W2)
            y = tf.add(out_layer, D_b2)
            # y = tf.nn.sigmoid(out_layer)

            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """

        # l = -tf.reduce_mean(tf.log(y) + tf.log(1. - y_hat))
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.ones_like(y)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))
        l = D_loss_real + D_loss_fake
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            if (reuse):
                scope.reuse_variables()
            # hidden_layer = tf.matmul(z, self.G_W1)
            # hidden_layer = tf.add(hidden_layer, self.G_b1)
            # hidden_layer = tf.nn.relu(hidden_layer)
            # out_layer = tf.matmul(hidden_layer, self.G_W2)
            # out_layer = tf.add(out_layer, self.G_b2)
            # x_hat = tf.nn.sigmoid(out_layer)
            G_W1 = tf.get_variable('G_W1', [self._nlatent, 128], initializer = layers.xavier_initializer(), trainable=True)
            G_b1 = tf.get_variable('G_b1', [128], initializer=tf.constant_initializer(0), trainable=True)

            G_W2 = tf.get_variable('G_W2', [128, self._ndims], initializer = layers.xavier_initializer(), trainable=True)
            G_b2 = tf.get_variable('G_b2', [self._ndims], initializer=tf.constant_initializer(0), trainable=True)

            hidden_layer = tf.matmul(z, G_W1)
            hidden_layer = tf.add(hidden_layer, G_b1)
            hidden_layer = tf.nn.relu(hidden_layer)
            out_layer = tf.matmul(hidden_layer, G_W2)
            out_layer = tf.add(out_layer, G_b2)
            x_hat = tf.nn.sigmoid(out_layer)

            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # l = -tf.reduce_mean(tf.log(y_hat))
        l = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))
        return l
