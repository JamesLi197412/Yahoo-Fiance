import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def feature_engineering(df,window_size):
    df['Gap'] = df['High'] - df['Low']
    df['Buying Pressure'] = np.where(df['Open'] > df['Close'], 1,0)

    # find out weekly Volume and conduct comparison
    df_weekly_volume = df.groupby(['Ticker','year','month','week'])['Volume'].mean().reset_index()
    df_weekly_volume.rename(columns ={'Volume':'Weekly Average Volume'},inplace = True)

    df = pd.merge(df, df_weekly_volume, how = 'inner', on =['Ticker','year','month','week'])
    df['Volume Gap(Positive)'] = np.where(df['Volume'] > df['Weekly Average Volume'],1,0)

    # Utilise moving average to conduct Trend analysis ( Adj Close)
    # Current Value > Previous Value, Positive, else Negative
    df['Adj Close Comp'] = df[f"Adj Close_mean_lag_{window_size}"].diff()
    df['Adj Close Comp'] = df['Adj Close Comp'].fillna(0)
    df['Adj Close Indicator'] = np.where(df['Adj Close Comp']>= 0, 1,0)

    return df


def monthly_aggregated(df):
    monthly_data = df.groupby(['Ticker','year', 'month'])['Volume','Close'].mean().reset_index()
    sorted_monthly_df = monthly_data.sort_values(by = ["year", "month"])
    sorted_monthly_df['date'] = pd.to_datetime(sorted_monthly_df[['year', 'month']].assign(DAY=1))
    print(sorted_monthly_df.head(10))
    return sorted_monthly_df

def monthly_plot(df):
    monthly_data = monthly_aggregated(df)
    sns.displot(data=monthly_data, x='Close',hue='Ticker', kde=True, bins=25, height=6, aspect=2.5).set(title='Stock closing prices (monthly avg)')
    g = sns.relplot(data=monthly_data, x='date', y='Close', hue='Ticker', kind='line', height=6, aspect=2.5)
    g.set(title='Stock closing prices (monthly avg)')
    g.set(xticks=[])
    plt.figure(figsize=(15, 7))
    g = sns.barplot(data=monthly_data, x='date', y='Volume', hue = 'Ticker')
    g.set(title='Stock volume (monthly avg) ')
    for index, label in enumerate(g.get_xticklabels()):
        if index % 24 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    g.tick_params(bottom=False)
    g.plot()
    plt.show()


