from analysis.data_exploration import *
from analysis.data_processing import *
from data.data_collect import data_collection
import warnings
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
warnings.simplefilter(action='ignore', category=FutureWarning)
from model.XGBoost import *
#from model.LSTM import *
#from model.ARIMA import *
import xgboost as xgb

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

def stock_analysis(STOCK_COMPANYS,INTERVAL,NUM_DAYS, windows_lag,window_size):
    # Call API to gather Data
    columns = ['Open','High','Close','Low','Adj Close','Ticker','Volume']
    stock_df = data_collection(STOCK_COMPANYS,INTERVAL,NUM_DAYS,columns, windows_lag)
    # Time-Series Analysis
    #exploration = Exploration()
    #stock_df_add =  exploration.df_exploration(stock_df)  # DataFrame Exploration
    stock_features_df = feature_engineering(stock_df, window_size)  # Feature Engineering
    #monthly_df =  monthly_aggregated(stock_df_add)
    #monthly_plot(stock_df_add)

    cols_drop = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Date','Ticker',
           'year', 'month', 'week', 'day', 'day_of_week', 'Open_mean_lag_7',
           'Open_std_lag_7', 'High_mean_lag_7', 'High_std_lag_7',
           'Close_mean_lag_7', 'Close_std_lag_7', 'Low_mean_lag_7',
           'Low_std_lag_7', 'Adj Close_mean_lag_7', 'Adj Close_std_lag_7',
           'Volume_mean_lag_7', 'Volume_std_lag_7']
    features_col = [ 'Gap', 'Buying Pressure', 'Volume Gap(Positive)','Adj Close Indicator']
    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }

    for ticker in STOCK_COMPANYS:
        ticker_df = stock_features_df[(stock_features_df["Ticker"] == ticker)].copy(deep = True)
        xgb.set_config(verbosity=0)
        xgboost_model = StockForecaster(ticker_df,features_col,'Close',parameters,cols_drop)
        xgboost_result = xgboost_model.model_run()
        return xgboost_result




    #return stock_features_df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define Global Parameters
    NUM_DAYS = 10000
    INTERVAL = '1d'
    windows_lag =7
    window_size = 7
    percentage_change = 0.1

    STOCK_COMPANYS = ['SPY', 'AAPL', 'AMD', 'NVDA']
    data = stock_analysis(STOCK_COMPANYS,NUM_DAYS, INTERVAL,windows_lag,window_size)
    #data.to_csv('test.csv')






