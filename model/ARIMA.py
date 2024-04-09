from data_processing import data_collection
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

# For non-seasonal data
def arima(p,q,d,attribute):
    #p=1, d=1, q=0 or 1
    df = data_collection('Apple')
    model=ARIMA(df[attribute],order=(p,d,q))
    model_fit=model.fit()
    model_fit.summary()

    df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
    df[[attribute,'forecast']].plot(figsize=(12,8))

    return df


def sarimax(df, p, q, d,attribute):
    ## SARIMAX
    model=sm.tsa.statespace.SARIMAX(df[attribute],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()
    df['forecast']=results.predict(start=90,end=103,dynamic=True)
    df[[attribute,'forecast']].plot(figsize=(12,8))

    future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

    future_datest_df.tail()

    future_df=pd.concat([df,future_datest_df])

    future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
    future_df[[attribute, 'forecast']].plot(figsize=(12, 8))

    return future_df