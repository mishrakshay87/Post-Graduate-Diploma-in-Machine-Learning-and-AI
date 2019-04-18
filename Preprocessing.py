import pandas as pd
import numpy as np
def preprocess_for_regularization(data,y_column_name,x_column_names):

    mean_list = {}
    std_list = {}
    y_column_name = "SalePrice"

    dataset_y=  data.loc[:,y_column_name]
    dataset_x=  data.loc[:,x_column_names]


    for col_name in x_column_names:
        mean_list[col_name]=(np.mean(dataset_x[col_name]))
        std_list[col_name]=(np.std(dataset_x[col_name],ddof=0))

    for col_name in x_column_names:
        dataset_x[col_name] = (dataset_x[col_name] - mean_list[col_name])/std_list[col_name]

    y_mean = np.mean(dataset_y)

    dataset_y = dataset_y - y_mean

    df1 = pd.DataFrame(dataset_x)
    df1[y_column_name] = dataset_y

    return df1
