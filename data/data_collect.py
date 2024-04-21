import datetime
import yfinance as yf
import os
import smtplib
from email.mime.text import MIMEText
import pandas as pd


def data_collection(STOCK_COMPANY, num_days, interval,features,windowSize):
    start = (datetime.date.today() - datetime.timedelta(num_days))
    end = datetime.datetime.today()
    df = pd.DataFrame(columns = ['Open','High','Close','Low','Adj Close','Ticker','Volume'])
    features =['Open','High','Close','Low','Adj Close','Volume']
    for stock in STOCK_COMPANY:
        data = single_stock_process(stock,start, end, interval,features,windowSize)
        data['Ticker'] = stock
        df = pd.concat([df,data], axis = 0)

    # df.to_csv('test.csv')
    # data_quality_check(df)
    return df

def single_stock_process(stock,start, end, interval,features,windowSize):
    data = yf.download(stock, start = start, end = end, interval = interval)
    data_features = date_info(data)
    data_lagged = window_leverage(data_features, windowSize, features)
    return data_lagged


def data_quality_check(df,missing_threshold = 0.7):
    # columns to discard if higher than threeshold
    missing_table = df.isnull().mean()
    columns_to_discard = missing_table[missing_table > missing_threshold].index.tolist()
    if (len(columns_to_discard) >= 1):
        # print(columns_to_discard)
        df = df.drop(columns_to_discard, axis = 1)
        message = 'Columns are dropped due to high missing portion'
        # send message
        email_notification(message)

def email_notification(messages):
    # For security reason, set your local environment
    sender_email = os.environ.get("SENDER_EMAIL")
    password = os.environ.get('EMAIL_PASSWORD')
    receiver_email = 'receiver@gmail.com'
    smtp_server = 'smtp.example.com'
    port = 587
    message = MIMEText(messages)
    message['Subject'] = 'Data Quality Check Alert'
    message['From'] = sender_email
    message['To'] = receiver_email
    with smtplib.SMTP_SSL(smtp_server,port) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email,receiver_email, message.as_string())

def date_info(df):
    df['Date'] = df.index
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df = df.set_index(df['Date'])
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df['week'] = df.Date.dt.week
    df['day'] = df.Date.dt.day
    df["day_of_week"] = df.Date.dt.dayofweek

    return df

def window_leverage(df, windowSize, features):
    for feature in features:
        df[f"{feature}_mean_lag_{windowSize}"]  = df[feature].rolling(windowSize).mean()
        df[f"{feature}_std_lag_{windowSize}"] = df[feature].rolling(windowSize).std()
        df[f"{feature}_mean_lag_{windowSize}"] = df[f"{feature}_mean_lag_{windowSize}"].fillna(0)
        df[f"{feature}_std_lag_{windowSize}"] = df[f"{feature}_std_lag_{windowSize}"].fillna(0)

    return df
