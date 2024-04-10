import datetime
import yfinance as yf

def data_collection(STOCK_COMPANY, num_days, interval):
    try:
        start = (datetime.date.today() - datetime.timedelta(days = num_days))
        end = datetime.datetime.today()
        data = yf.download(STOCK_COMPANY, start = start, end = end, interval = interval)
        print(data)
        data.rename(columns={'Close': 'close', 'High': 'high', "Low": 'low', 'Volume': 'volumne',
                            "Open": 'open'}, inplace=True)
        return data
    except Exception as e:
        print(f"Error:{e}")

    #data['date'] = data.index
    # Generate Dataframe for data


def data_quality_check(df):
    return None
