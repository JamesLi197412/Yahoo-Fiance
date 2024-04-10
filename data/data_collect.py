import datetime
import yfinance as yf


def download_stock(symbols, num_days, interval) -> DataFrame:
    """
        Download
    :param symbols:
    :param num_days:
    :param interval:
    :return:
    """
    start = (datetime.date.today() - datetime.timedelta(num_days))
    end = datetime.datetime.today()
    pdf = yf.download(symbols,start = start, end = end, interval = interval)
    pdf.rename(columns = {'Close':'close', 'High':'high',"Low":'low','Volume':'volumne',
                          "Open":'open'},inplace = True)

    pdf['date'] = pdf.index

    return pdf