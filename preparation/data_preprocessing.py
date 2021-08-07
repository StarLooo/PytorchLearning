# -*- coding: utf-8 -*-
import torch
import tensorflow as tf
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    data = pd.read_csv(data_file)
    print(data)
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs.fillna(inputs.mean(numeric_only=True), inplace=True)
    print(inputs)
    print(outputs)
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)
    # torch version
    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X)
    print(y)
    # tf version
    X, y = tf.constant(inputs.values), tf.constant(outputs.values)
    print(X)
    print(y)
