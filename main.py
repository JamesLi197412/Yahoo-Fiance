from analysis.data_exploration import *
from analysis.data_processing import *
from data.data_collect import data_collection
import warnings
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
warnings.simplefilter(action='ignore', category=FutureWarning)

def single_stock(stockList):
    # create ticker for Apple Stock
    #ticker = yf.Ticker('AAPL')
    #df = ticker.history(period='3Y')

    # Call the Object
    #dfexploration = Exploration()
    # window_lag = 7
    # Columns : Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Stock Splits',
    #        'Date', 'month', 'week', 'day', 'day_of_week'
    # features = ['Close','High','Low','Open','Volume']

    # df = dfexploration.window_Lag(df,7,features)

    #df = dfexploration.date_info(df)

    #dfexploration.plot_df(df,df['Date'],features, xlabel = 'Date',dpi = 100)
    # Export the Decomposition figure to the output folder
    #dfexploration.decomposition(df, 'Low', window_lag)

    # Go Check signal stationary or not
    #ts = df.loc[:,['Close']].copy(deep = True)
    #ts = ts.dropna()
    #ts.index = ts.Dates
    #ts = ts.drop("Date", axis = 1)
    # dfexploration.check_adfuller(ts)
    #result = dfexploration.test_stationarity(ts,'Close')
    #print(f'This Signal is {result}.')
    return None

def stock_analysis(STOCK_COMPANYS,INTERVAL,NUM_DAYS, windows_lag):
    # Call API to gather Data
    columns = ['Open','High','Close','Low','Adj Close','Ticker','Volume']
    stock_df = data_collection(STOCK_COMPANYS,INTERVAL,NUM_DAYS,columns, windows_lag)

    # Time-Series Analysis
    #exploration = Exploration()
    #stock_df_add =  exploration.df_exploration(stock_df)  # DataFrame Exploration
    test_df = feature_engineering(stock_df, window_size)
    #monthly_df =  monthly_aggregated(stock_df_add)
    #monthly_plot(stock_df_add)
    # monthly_plot(stock_df_add)
    # for feature in features:
    test = ['Open','High','Close','Low','Adj Close']
    # stock_df_window = exploration.window_Lag(stock_df_add, windows_lag, test)



    return test_df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define Global Parameters
    NUM_DAYS = 10000
    INTERVAL = '1d'
    windows_lag = 7
    window_size = 30
    percentage_change = 0.1

    STOCK_COMPANYS = ['SPY', 'AAPL', 'AMD', 'NVDA']
    #STOCK_COMPANYS = ['AMD', 'NVDA']
    data = stock_analysis(STOCK_COMPANYS,NUM_DAYS, INTERVAL,windows_lag)

    data.to_csv('test.csv')






