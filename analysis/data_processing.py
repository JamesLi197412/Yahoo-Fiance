import yfinance as yf
from model.LSTM import StockLSTM
from data.data_collect import data_collection
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

import seaborn as sns


def feature_engineering(df,window_size):
    df['Gap'] = df['High'] - df['Low']
    df['Buying Pressure'] = np.where(df['Open'] > df['Close'], 1,0)

    # find out weekly Volume and conduct comparison
    df_weekly_volume = df.groupby(['Ticker','year','month','week'])['Volume'].mean().reset_index()
    #df_weekly_volume['Volume'] = df_weekly_volume['Volume'].fillna(0)
    df_weekly_volume.rename(columns ={'Volume':'Weekly Average Volume'},inplace = True)

    df = pd.merge(df, df_weekly_volume, how = 'inner', on =['Ticker','year','month','week'])
    df['Volume Gap(Positive)'] = np.where(df['Volume'] > df['Weekly Average Volume'],1,0)

    # Utilise moving average to conduct Trend analysis

    return df


def monthly_aggregated(df):
    monthly_data = df.groupby(['Ticker','year', 'month'])['Volume','Close'].mean().reset_index()
    sorted_monthly_df = monthly_data.sort_values(by = ["year", "month"])
    sorted_monthly_df['date'] = pd.to_datetime(sorted_monthly_df[['year', 'month']].assign(DAY=1))
    print(sorted_monthly_df.head(10))
    return sorted_monthly_df

def monthly_plot(df):
    monthly_data = monthly_aggregated(df)
    sns.displot(data=monthly_data, x='Close',hue='Ticker', kde=True, bins=25, height=6, aspect=2.5).set(title='Stock closing prices (monthly avg)')
    g = sns.relplot(data=monthly_data, x='date', y='Close', hue='Ticker', kind='line', height=6, aspect=2.5)
    g.set(title='Stock closing prices (monthly avg)')
    g.set(xticks=[])
    plt.figure(figsize=(15, 7))
    g = sns.barplot(data=monthly_data, x='date', y='Volume', hue = 'Ticker')
    g.set(title='Stock volume (monthly avg) ')
    for index, label in enumerate(g.get_xticklabels()):
        if index % 24 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    g.tick_params(bottom=False)
    g.plot()
    plt.show()

def data_preprocessing(brandList, starttime,endtime,split_ratio,step, features):
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



