"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.

        # ele1 = np.zeros((f.shape[0], 1))
        y = y.reshape(f.shape[0],1)
        # print('f', f.shape)
        # print('y', y.shape)
        # print('f', f)
        z = np.multiply(f, y)
        # print('z',z)
        ele2 = 1 - z
        # print(ele2)
        # hinge_loss = np.sum(np.maximum(ele1, ele2))
        flag = np.zeros((f.shape[0], 1))
        for i in range(0,f.shape[0]):
            if ele2[i] > 0:
                flag[i] = 1
        # print('flag', flag)
        y = flag * y
        loss_grad = - np.matmul(self.x.T, y)

        reg_grad = self.w_decay_factor * self.w
        total_grad = reg_grad + loss_grad
        # print('totad_ grad', total_grad)

        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """

        hinge_loss = None
        l2_loss = None
        # Implementation here.
        y = y.reshape(y.shape[0], 1)
        ele1 = np.zeros((f.shape[0], 1))
        z = np.multiply(f, y)
        ele2 = 1 - z
        hinge_loss = np.sum(np.maximum(ele1, ele2))
        l2_loss = self.w_decay_factor/2 * np.linalg.norm(self.w)**2
        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,). Tie break 0 to 1.0.
        """
        y_predict = None
        # Implementation here.
        y_predict = []
        for i in f:
            if i < 0:
                y_predict.append(-1)
            else:
                y_predict.append(1)
        y_predict = np.array(y_predict).reshape(f.shape[0],1)


        return y_predict
