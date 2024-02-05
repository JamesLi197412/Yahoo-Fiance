import datetime
import yfinance as yf
from exploration import Exploration
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(stockList):
    # create ticker for Apple Stock

    dfexploration = Exploration()

    ticker = yf.Ticker('AAPL')
    todays_data = ticker.history(period='3Y')

    dfexploration.df_exploration(todays_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    stockList = ['AAPL','IBM']
    main(stockList)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
