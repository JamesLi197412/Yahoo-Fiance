import yfinance as yf
from model.LSTM import StockLSTM
from data.data_collect import data_collection
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn


def feature_engineering(df,window_size):
    df['gap'] = df['high'] - df['low']
    df['buying pressure'] = np.where(df['open'] > df['close'], 1,0)

    # find out monthly weekly Volumn and conduct comparison

    # Utilise moving average to conduct Trend analysis

    return df


def monthly_aggregated(df):
    return df


def data_processing(brandList, starttime,endtime,split_ratio,step, features):
    data = data_collection(brandList, starttime, endtime)

    # Split the data into train & test
    train_df, test_df, train_size = split_data(data,split_ratio)

    # Generate Dataset
    X_train, y_train =  create_dataset(train_df,step,features[0])
    X_test, y_test = create_dataset(test_df, step, features[0])

    return X_train, y_train, X_test, y_test, train_size

def split_data(df,ratio):
    train_size = int(len(df) * ratio)
    test_size = len(df) - train_size

    train,test = df[:train_size], df[train_size :]
    return train, test, train_size

def create_dataset(df,lookback, attribute):
    df = df[[attribute]].values
    X, y = [], []
    for i in range(len(df) - lookback):
        feature = df[i:i+lookback]
        target = df[i+1: i+lookback+1]
        X.append(feature)
        y.append(target)

    #return X,y
    return torch.Tensor(X),torch.Tensor(y)

def forecast(X_train,y_train,X_test,y_test, lookback,train_size,ts):
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
    plt.plot(ts)
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()



