
from analysis.data_exploration import Exploration
from data.data_collect import data_collection
import warnings

import matplotlib
matplotlib.use('TkAgg')



warnings.simplefilter(action='ignore', category=FutureWarning)

def single_stock(stockList):
    # create ticker for Apple Stock
    #ticker = yf.Ticker('AAPL')
    #df = ticker.history(period='3Y')

    # Call the Object
    #dfexploration = Exploration()
    window_lag = 7
    # Columns : Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Stock Splits',
    #        'Date', 'month', 'week', 'day', 'day_of_week'
    features = ['Close','High','Low','Open','Volume']

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

def stock_analysis(STOCK_COMPANY,INTERVAL, NUM_DAYS):
    data = data_collection(STOCK_COMPANY,INTERVAL,NUM_DAYS)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define Global Parameters
    NUM_DAYS = 10000
    INTERVAL = '1d'
    # List of Stock Companys
    STOCK_COMPANY = ['SPY', 'AAPL', 'AMD', 'NVDA']

    warnings.simplefilter(action='ignore', category=FutureWarning)
    Test = 'IBM'
    data = stock_analysis(Test,INTERVAL, NUM_DAYS)
    print(data)
    #stockList = ['AAPL','IBM']
    # single_stock(stockList)



