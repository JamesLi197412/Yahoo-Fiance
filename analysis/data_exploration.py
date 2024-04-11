import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot

class Exploration:
    def df_exploration(self, df):
        """
            Information about the df
            # Columns Data Type
            # Data Frame shape
            # Columns Name
            # Columns Description
            # First 5 Data Samples
        """
        features_dtypes = df.dtypes
        rows, columns = df.shape

        missing_values_cols = df.isnull().sum()
        missing_col = missing_values_cols.sort_values(ascending=False)
        features_names = missing_col.index.values
        missing_values = missing_col.values

        print('=' * 50)
        print('===> This data frame contains {} rows and {} columns'.format(rows, columns))
        print('=' * 50)

        print("{:13}{:13}{:30}".format('Feature Name'.upper(),
                                            'Data Format'.upper(),
                                            'The first five samples'.upper()))

        for features_names, features_dtypes in zip(features_names, features_dtypes[features_names]):
            print('{:15} {:14} '.format(features_names, str(features_dtypes)),end=" ")

            for i in range(5):
                print(df[features_names].iloc[i], end=",")

            print("=" * 50)

    def date_info(self,df):
        """
            Add month, week, day and day of week features in prepartion for model
        :param df: Signal
        :return: Signal with date features
        """
        df['Date'] = df.index
        df.Date = pd.to_datetime(df.Date, format = "%Y-%m-%d")
        df = df.set_index(df['Date'])
        df['month'] = df.Date.dt.month
        df['week'] = df.Date.dt.week
        df['day'] = df.Date.dt.day
        df["day_of_week"] = df.Date.dt.dayofweek

        return df

    def window_Lag(self,df,windowSize,features):
        dfRolledLags =df[features].rolling(window = windowSize, min_periods = 0)

        dfMeanLags = dfRolledLags.mean().shift(1).reset_index()
        dfStdLags = dfRolledLags.std().shift(1).reset_index()
        for feature in features:
            df[f"{feature}_mean_lag{windowSize}"] = dfMeanLags[feature]
            df[f"{feature}_std_lag{windowSize}"] = dfStdLags[feature]

        df.fillna(df.mean(), inplace = True)
        return df

    # Moving average for 7
    def moving_average(self,data, window, attribute):
        data['Moving Average' + attribute + ' ' + str(window)] = data[attribute].rolling(window).mean()
        actual = data[attribute][-(window + 7):]
        ma = data['Moving Average' + attribute + str(window)][-(window + 7):]

        plt.figure(figsize=(20, 8))
        actual.plot(label='Actual', lw=4)
        ma.plot(label='MA-{}'.format(str(window)), ls='--', lw=2)
        plt.title('{}-Days Moving Average'.format(str(window)), weight='bold', fontsize=25)
        plt.legend()

    # Time series plot by attributes
    def plot_df(self,df, x,features,xlabel, dpi):
        loop = len(features)
        fig, axes = plt.subplots(nrows = loop, ncols = 1, figsize = (200,150), dpi = dpi)
        fig.suptitle('Stock Price Features in time domain')
        for i in range(loop):
            y = df[features[i]]
            axes[i].plot(x, y, color='tab:red')
            axes[i].set(xlabel = xlabel, ylabel = features[i])

        fig.tight_layout()
        plt.show()
        plt.savefig('output/Stock Signal Features.png')

    def decomposition(self,df,yValue, windowSize):
        """
           Time series data decompostion to visualise its trend, seasonal, Residuals
        :param df: Signals
        :param yValue: stock values
        :param windowSize: period
        :return:
        """
        # Multiplicative Decomposition
        df = df.dropna()
        multiplicative_decomposition = seasonal_decompose(df[yValue],model='multiplicative',period=windowSize)

        # Additive Decomposition
        additive_decomposition = seasonal_decompose(df[yValue], model='additive',period=windowSize)

        # Plot
        plt.rcParams.update({'figure.figsize': (16, 12)})
        multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('output/visualisations/Multiplicative Decomposition.png')

        additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('output/visualisation/Additive Decomposition.png')
        plt.show()


    #  Trend Stationary -- A time series that does not show a trend
    # Seasonal Stationary -- A time series that does not show seasonal changes
    # Strictly Stationary -- The joint distribution of observations is invariant to time shift.
    def check_adfuller(self,ts):
        """
            Augmented Dickey - Fuller test is a type of statistical test called a unit root test, which determines how strongly a time series is defined by a trend.
            The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary(has some time-dependent structure). The alternatvie hypothesis (rejecting the nul hypothesis) is that the  time series is stationary.
            1. Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-sataionary.
            2. Alternative Hypothesis(H1): The null hypothesis is rejected; it suggestes the time series does not have a unit root.

            based on its AFD test and p value to judge its stationary or not,
            Conditions:
            a)     if p <= 0.05, then reject the null hypothesis --> The signal is stationary.
            b)     ADF statistics value higher than any critical values

            ADF Statistic: The more negative of test statistic value, the more likely we are to reject the null hypothesis.

            IF ADF Statistics value is less than the value of 1% value from Critical Values, then it suggests that we can reject the null hypothesis with a significance level of less than 1%(i.e. aa low probability that the result is a statistical fluke)

            Rejecting the null hypothesis means that the process has no unit, and in turn that the time series is stationary or does not have time-dependent structure.

            IF ADF Statistics Test is higher than any critical values, it means that we can fail to reject the null hypothesis and in turn that the time series is non-stationary.
        :param ts:
        :return: ADF Test Result
        """
        # Dickey-Fuller test
        result = adfuller(ts, autolag='AIC')
        print('Test statistic: ', result[0])
        print('p-value: ', result[1])
        print('Critical Values:', result[4])

        return result

    def test_stationarity(self,ts,ylabel):
        """
            Present the ADF Test Result so that checking signal stationary or not
            Augmented Dickey-Fuller test
        :param df:
        :return:
        """

        results = self.check_adfuller(ts)
        if (results[0] > results[4]['1%'] and results[1] < 0.5):
            assumes = 'Not Stationary'
        else:
            assumes = 'Stationary'
        return assumes


    def autocorrelation_correlation(self,df,col):
        # Autocorrelation to check seasonality
        plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
        autocorrelation_plot(df[col])
        plt.title('Autocorrelation', fontsize=16)
        plt.plot()

        plot_acf(df[col], ax=plt.gca(), lags=150)
        plt.show()

        plot_pacf(df[col], ax=plt.gca(), lags=100)
        plt.show()