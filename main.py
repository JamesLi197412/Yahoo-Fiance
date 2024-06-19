from analysis.data_exploration import *
from analysis.data_processing import *
from data.data_collect import data_collection
import warnings
import matplotlib
matplotlib.use('TkAgg')
warnings.simplefilter(action='ignore', category=FutureWarning)
from model.XGBoost import *
from model.LSTM import *
from model.ARIMA import *



def arima_operation(df,attribute):
    ts = df.copy()
    ts.index = ts['Date']
    ts = ts.loc[:, [attribute]].copy(deep=True)
    ts = ts.fillna(0)
    order = (0,1,0)
    arima = ARIMA_Model(ts,order = order)
    arima_prediction = arima.run()

    return arima_prediction


def xgboost_operation(df,parameters):
    cols_drop = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Date','Ticker',
           'year', 'month', 'week', 'day', 'day_of_week', 'Open_mean_lag_7',
           'Open_std_lag_7', 'High_mean_lag_7', 'High_std_lag_7',
           'Close_mean_lag_7', 'Close_std_lag_7', 'Low_mean_lag_7',
           'Low_std_lag_7', 'Adj Close_mean_lag_7', 'Adj Close_std_lag_7',
           'Volume_mean_lag_7', 'Volume_std_lag_7']
    features_col = [ 'Gap', 'Buying Pressure', 'Volume Gap(Positive)','Adj Close Indicator']


    #for ticker in STOCK_COMPANYS:
    #    ticker_df = df[(df["Ticker"] == ticker)].copy(deep = True)
    #    xgb.set_config(verbosity=0)
    #    xgboost_model = StockForecaster(ticker_df,features_col,'Close',parameters,cols_drop)
    #    xgboost_result = xgboost_model.model_run()
    return df

def lstm_operation(df,attribute):
    ts = df.copy()
    ts.index = ts['Date']
    ts = ts.loc[:, [attribute]].copy(deep=True)
    ts = ts.fillna(0)
    xgboost_model = LSTM_Model(ts)
    xgboost_prediction = xgboost_model.run(time_step= 2)
    return xgboost_prediction


def stock_analysis(STOCK_COMPANYS,INTERVAL,NUM_DAYS, windows_lag,window_size):
    # Call API to gather Data
    columns = ['Open','High','Close','Low','Adj Close','Ticker','Volume']
    stock_df = data_collection(STOCK_COMPANYS,INTERVAL,NUM_DAYS,columns, windows_lag)

    # Time-Series Analysis
    #exploration = Exploration()
    #stock_df_add =  exploration.df_exploration(stock_df)  # DataFrame Exploration
    stock_features_df = feature_engineering(stock_df, window_size)  # Feature Engineering
    #print(stock_features_df.columns)

    # Monthly Aggregated Data
    #monthly_df =  monthly_aggregated(stock_df_add)
    #monthly_plot(stock_df_add)

    # Predict flows with 3 various algorithms
    arima_prediction = arima_operation(stock_features_df, 'Close')

    lstm_parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }
    lstm_prediction = lstm_operation(stock_features_df,'Close')

    xgboost_prediction = xgboost_operation(stock_features_df,lstm_parameters)


    return arima_prediction, xgboost_prediction, lstm_prediction


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define Global Parameters
    NUM_DAYS = 10000
    INTERVAL = '1d'
    windows_lag =7
    window_size = 7
    percentage_change = 0.1

    # STOCK_COMPANYS = ['SPY', 'AAPL', 'AMD', 'NVDA']
    STOCK_COMPANYS = ['SPY']
    arima_prediction, xgboost_prediction, lstm_prediction = stock_analysis(STOCK_COMPANYS,NUM_DAYS, INTERVAL,windows_lag,window_size)








