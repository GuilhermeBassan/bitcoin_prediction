# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:54:31 2020

@author: guilherme
"""

import time
start = time.time()

# Importing libraries
import json
import requests
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Canadian exchange rate stored into pandas dataFrame
# to_datetime() is used to convert string datetime to python datetime object
# This is necessary to perform operations of time difference

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=BRL&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit = 's')
target_col = 'close'

# Show five first ines
#print(hist.head(5))

# Function to split the data sets
def train_test_split(df, test_size = 0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

# Splitting data into two sets - 80% training and 20% testing
train, test = train_test_split(hist, test_size = 0.2)

# Plotting the cryptoC in BRL as function of time
def line_plot(line1, line2, label1 = None, label2 = None,
              title = '', lw = 2):
    fig, ax = plt.subplots(1, figsize = (13,7))
    ax.plot(line1, label = label1, linewidth = lw)
    ax.plot(line2, label = label2, linewidth = lw)
    ax.set_ylabel('PREÇO [BRL]', fontsize = 14)
    ax.set_title(title, fontsize = 16)
    ax.legend(loc = 'best', fontsize = 16)
    
#line_plot(train[target_col], test[target_col], 'training', 'test', title = '')

# Normalization of the values
def normalize_zero_base(df):
    return df / df.iloc[0] - 1

def normalize_min_max(df):
    return (df - df.min()) / (data.max() - df.min())

# Extracting data from windows with size 5 each
def extract_window_data(df, window_len = 5, zero_base = True):
    window_data = []
    for idx in range (len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalize_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

# Function to prepare the data in a format to be fed to the NN
def prepare_data(df, target_col, window_len = 10, zero_base = True,
                 test_size = 0.2):
    train_data, test_data = train_test_split(df, test_size = test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    Y_train = train_data[target_col][window_len:].values
    Y_test = test_data[target_col][window_len:].values
    if zero_base:
        Y_train = Y_train / train_data[target_col][: - window_len].values - 1
        Y_test = Y_test / test_data[target_col][: - window_len].values - 1
    return train_data, test_data, X_train, X_test, Y_train, Y_test

# LSTM - Long Short Term Memory
# A recurrent network, takes information from both previous layers and the
# current layer. Can remember important information and forget irrelevant info

# The NN comprises a LSTM layer followed by a 20% Dropout layer and a Dense 
# layer with ReLU function. 
def build_lstm_model(input_data, output_size, neurons = 100, 
                     activ_func = 'linear', dropout = 0.2, 
                     loss = 'mse', optimizer = 'adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape = (input_data.shape[1], 
                                           input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units = output_size))
    model.add(Activation(activ_func))
    model.compile(loss = loss, optimizer = optimizer)
    return model

# Parameters to be used later
np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 1000
epochs = 100
batch_size = 50
loss = 'mse'
dropout = 0.35
optimizer = 'adam'

# Training the model
(train, test, X_train, X_test,
 Y_train, Y_test) = prepare_data(hist, target_col, window_len = window_len,
                                 zero_base = zero_base, test_size = test_size)
                                 
model = build_lstm_model(X_train, output_size = 1, neurons = lstm_neurons, 
                         dropout = dropout, loss = loss, optimizer = optimizer)

history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size,
                    verbose = 1, shuffle = True)

# Mean Absolute Error (mse)
targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, Y_test) 

# Plot the results
preds = test[target_col].values[: - window_len] * (preds + 1)
preds = pd.Series(index = targets.index, data = preds)
line_plot(targets, preds, 'VERDADEIRO', 'PREDIÇÃO', lw = 3)

finish = time.time()

print("Finished in", round((finish - start), 2), "seconds")




















