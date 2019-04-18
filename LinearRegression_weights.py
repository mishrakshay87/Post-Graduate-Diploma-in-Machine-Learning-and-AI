import pandas as pd
import numpy as np


def least_squares_weights(input_x, target_y):
    row_x, col_x = input_x.shape
    row_y, col_y = target_y.shape

    if row_x < col_x:
        input_x = np.transpose(input_x)

    row_x, col_x = input_x.shape

    ones_mat = np.ones((row_x, 1))
    input_x = np.hstack((ones_mat, input_x))

    weights_temp = np.linalg.inv(np.matmul(np.transpose(input_x), input_x))

    weights = np.matmul(np.transpose(input_x), target_y)

    weights = np.matmul(weights_temp, weights)

    return weights