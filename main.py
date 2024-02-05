


import yfinance as yf
def main():
    # create ticker for Apple Stock
    ticker = yf.Ticker('AAPL')
    # get data of the most recent date
    todays_data = ticker.history(period='1d')

    print(todays_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
