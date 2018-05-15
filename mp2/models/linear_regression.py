"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model."""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,1).
        """

        total_grad = np.matmul(self.x.T, f-y) + self.w_decay_factor * self.w

        return total_grad

    def total_loss(self, f, y):
        """Computes the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """
        total_loss = None

        squared_loss = 0.5 * np.linalg.norm(f-y)**2
        # print(squared_loss)
        # print(1)
        regularization = self.w_decay_factor * 0.5 * np.linalg.norm(self.w)**2
        total_loss = squared_loss + regularization

        return total_loss

    def predict(self, f):
        """Nothing to do here.
        """
        return f
