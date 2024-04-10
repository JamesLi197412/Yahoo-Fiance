from analysis.data_processing import data_collection
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

class ARIMA_Model:
  def __init__(self, data, order=(p, q, d)):
    self.data = data
    self.order = order
    self.model = None

  def _check_model_built(self):
    if self.model is None:
      raise Exception("Model not yet built. Please call build_model() first.")

  def build_model(self):
    """
    Builds and fits an ARIMA model to the data.
    """
    self.model = ARIMA(self.data, order=self.order)
    self.model = self.model.fit()

  def forecast(self, steps):
    """
    Generates forecasts for the given number of steps using the fitted model.
    steps (int): The number of steps to forecast.
    """

    self._check_model_built()
    forecast_values = self.model.forecast(steps=steps)
    return forecast_values

  def evaluate(self, metric):
    """
    Evaluates the fitted model using a specified metric.

    Args:
        metric (callable): A function that takes the actual and forecast values as arguments
                             and returns a performance metric.

    Returns:
        float: The evaluation metric score.
    """

    self._check_model_built()
    # Implement logic to calculate the chosen metric here (e.g., RMSE, MAE)
    # This example uses a placeholder function
    return metric(self.data, self.model.forecast(steps=len(self.data)))


def arima_execute(data):
    order = (1,2,1)
    model = ARIMA_Model(data,order = order)
    model.build_model()
    print(model.model.summary())  # Print model summary

    forecast_steps = 10
    forecasts = model.forecast(forecast_steps)

    # Plot actual data and forecasts
    data.plot(label="Actual")
    forecasts.plot(label=f"Forecast ({forecast_steps} steps)")

# Example evaluation (replace with your desired metric function)
def mean_squared_error(actual, forecast):
  return ((actual - forecast) ** 2).mean()

evaluation_score = model.evaluate(mean_squared_error)
print(f"Evaluation Score (MSE): {evaluation_score}")


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