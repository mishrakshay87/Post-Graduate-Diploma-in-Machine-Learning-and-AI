import pandas as pd
import numpy as np
def ridge_regression_weights(input_x, output_y, lambda_param):

    row_x,col_x =  input_x.shape

    if row_x < col_x:
        input_x = np.transpose(input_x)

    row_x,col_x =  input_x.shape

    append_one = np.ones((row_x,1))

    Xnew = np.hstack((append_one,input_x))

    ident = np.eye(col_x+1)
    ident = ident * lambda_param
    x_tran = np.matmul(np.transpose(Xnew),Xnew)

    mat_sum = ident + x_tran

    inv_mat = np.linalg.inv(mat_sum)

    weight_temp = np.matmul(np.transpose(Xnew),output_y)

    weight_fin = np.matmul(inv_mat, weight_temp)
    return weight_fin