import datetime
import yfinance as yf
import os
import smtplib
from email.mime.text import MIMEText

def data_collection(STOCK_COMPANY, num_days, interval):
    try:
        start = (datetime.date.today() - datetime.timedelta(num_days))
        end = datetime.datetime.today()
        data = yf.download(STOCK_COMPANY, start = start, end = end, interval = interval)
        data.rename(columns={'Close': 'close', 'High': 'high', "Low": 'low', 'Volume': 'volumne',
                            "Open": 'open'}, inplace=True)
        data['Brand'] = STOCK_COMPANY
        data_quality_check(data, 0.7)
        return data
    except Exception as e:
        print(f"Error:{e}")


def data_quality_check(df,missing_threshold):
    # columns to discard if higher than threeshold
    missing_table = df.isnull().mean()
    columns_to_discard = missing_table[missing_table > missing_threshold].index.tolist()
    if (len(columns_to_discard) >= 1):
        # print(columns_to_discard)
        df = df.drop(columns_to_discard, axis = 1)
        message = 'Columns are dropped due to high missing portion'
        # send message
        email_notification(message)

    # Check for Incorrect High/Low Relationship
    subset = (df[df['low'] > df['high']])
    if (subset.shape[0] >=1):
        message = "Some rows low & high does not look logically."
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