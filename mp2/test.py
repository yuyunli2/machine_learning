"""Simple unit tests for students."""

import unittest
import numpy as np
from utils import io_tools
from models import linear_regression


class IoToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset("data/train.csv")

    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_first_row(self):
        # keys = list(self.dataset.keys())
        # print(keys)
        keys = sorted(list(self.dataset.keys()))
        val0 = (self.dataset[keys[0]])
        val0_true = ('1', '1Fam', '7', '1710', '548', '208500')
        self.assertEqual(val0, val0_true)

    # def test_process_data(self):
        # data = 
# class ModelTests2(unittest.TestCase):
#     def setUp(self):
#         self.model = linear_regression.LinearRegression(5, 'uniform')
        # print(self.w)

    # def test_forward_shape(self):
    #     print(self.w)
    #

class ModelTests(unittest.TestCase):
    def setUp(self):
        # self.model = linear_regression.LinearRegression(5, 'uniform')

        self.model = linear_regression.LinearRegression(5, 'zeros')

    def test_forward_shape(self):
        x = np.zeros((10, 5))
        y_hat = self.model.forward(x)
        self.assertEqual(y_hat.shape, (10, 1))

    def test_forward_zero(self):
        x = np.zeros((10, 5))
        y = np.zeros((10, 1))
        y_hat = self.model.forward(x)
        np.testing.assert_array_equal(y, y_hat)

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.model = linear_regression.LinearRegression(5, 'uniform')

    def test_loss_shape(self):
        x = np.zeros((1,1))
        # y_hat = self.model.total_loss(x,x)
        # np.testing.assert_array_equal(x,y_hat)

    def test_backward(self):
        # print('back')
        x = np.zeros((10, 5))
        y_hat = self.model.forward(x)
        y = np.zeros((10,1))
        back = self.model.backward(y_hat,y)
        np.testing.assert_array_equal(back.shape, (6,1))

if __name__ == '__main__':
    unittest.main()
