import datetime
import yfinance as yf
from exploration import Exploration
import warnings
import pandas as pd
from model.LSTM import StockLSTM

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn



def data_collection():
    ticker = yf.Ticker('AAPL')
    df = ticker.history(period = '3Y')
    return df

def split_data(df):
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size

    train,test = df[:train_size], df[train_size :]
    return train, test

def create_dataset(df,lookback):
    df = df[['Close']].values
    X, y = [], []
    for i in range(len(df) - lookback):
        feature = df[i:i+lookback]
        target = df[i+1: i+lookback+1]
        X.append(feature)
        y.append(target)

    #return X,y
    return torch.Tensor(X),torch.Tensor(y)



def forecast():
    df = data_collection()
    ts = df.loc[:, ['Close']].copy(deep=True)
    ts = ts.dropna()
    train_size = int(len(ts) * 0.8)

    lookback = 7
    train, test = split_data(ts)
    X_train, y_train = create_dataset(train, lookback = lookback)
    X_test, y_test = create_dataset(test, lookback = lookback)

    print('Start Forecasting')

    model = StockLSTM()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train,y_train),shuffle = True, batch_size = 8)

    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()

        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(ts) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(ts) * np.nan
        test_plot[train_size + lookback:len(ts)] = model(X_test)[:, -1, :]

    # plot
    plt.plot(df)
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()



