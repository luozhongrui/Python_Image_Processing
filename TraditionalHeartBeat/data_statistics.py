#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 19:24:39
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from math import sqrt
import csv


def get_true():
    csvdata = []
    file = open('./true_data.xlsx.csv', 'r', encoding='utf-8')
    temp = csv.reader(file)
    for i in temp:
        csvdata.append(i[0])
    csvdata.pop(0)
    for i in range(len(csvdata)):
        csvdata[i] = float(csvdata[i])

    return csvdata


def get_predict():
    csvdata = []
    file = open('./predict.csv', 'r', encoding='utf-8')
    temp = csv.reader(file)
    for i in temp:
        csvdata.append(i[0])
    csvdata.pop(0)
    for i in range(len(csvdata)):
        csvdata[i] = float(csvdata[i])

    return csvdata


if __name__ == '__main__':
    ture_data = get_true()
    predict = get_predict()
    # print(ture_data, '\n', predict)
    true = np.array(ture_data)
    predict = np.array(predict)
    mae = mean_absolute_error(y_true=true, y_pred=predict)
    mse = mean_squared_error(y_true=true, y_pred=predict)
    rmse = sqrt(mse)
    corr = pearsonr(x=true, y=predict)
    x = abs(predict-true)
    print(max(x))
    print(x)
    print("MAE:", mae)
    print("CORR:", corr)
    print("MSE:", mse)
    print("RMSE:", rmse)
    plt.hist(x, density=True)
    plt.show()
