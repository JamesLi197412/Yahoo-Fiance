import datetime
import yfinance as yf
from exploration import Exploration
import warnings
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(stockList):
    # create ticker for Apple Stock
    ticker = yf.Ticker('AAPL')
    df = ticker.history(period='3Y')

    # Call the Object
    dfexploration = Exploration()
    # Data Process
    #df = dfexploration.date_info(df)
    window_lag = 7
    # Columns : Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Stock Splits',
    #        'Date', 'month', 'week', 'day', 'day_of_week'
    features = ['Close','High','Low','Open','Volume']

    #dfexploration.plot_df(df,df['Date'],features, xlabel = 'Date',dpi = 100)
    # Export the Decomposition figure to the output folder
    #dfexploration.decomposition(df, 'Low', window_lag)

    df = dfexploration.window_Lag(df,7,features)

    # Go Check signal stationary or not





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    stockList = ['AAPL','IBM']
    main(stockList)


