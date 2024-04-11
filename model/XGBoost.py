import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


class StockForecaster:
  """
  A class for loading, preprocessing, training, and evaluating an XGBoost model
  for stock price forecasting.
  """

  def __init__(self):
    self.data = None
    self.features = None
    self.target = None
    self.model = None


  def preprocess_data(self, target_col="Close"):
    """
    Performs feature engineering and defines target variable.

    Args:
        target_col (str, optional): The column name containing the target variable (e.g., closing price). Defaults to "Close".
    """

    # Feature engineering (add relevant technical indicators or other features)
    # ...

    self.target = self.data[target_col]
    self.features = self.data.drop(target_col, axis=1)

  def create_lagged_dataset(self, n_past=1):
    """
    Creates a dataset for XGBoost with lagged features.

    Args:
        n_past (int, optional): The number of past periods to use as features. Defaults to 1.

    Returns:
        pandas.DataFrame: The dataset with lagged features and target.
    """

    lagged_df = self.data.copy()
    for i in range(1, n_past + 1):
      lagged_df[f"lag_{i}"] = lagged_df.shift(i)[self.target.name]
    lagged_df.dropna(inplace=True)

    X = lagged_df.drop(self.target.name, axis=1)
    y = lagged_df[self.target.name]

    return X, y

  def train_forecast_evaluate(self, X_train, y_train, X_test, n_forecast=1):
    """
    Trains an XGBoost model, forecasts future values, and evaluates performance.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.
        X_test (pandas.DataFrame): Testing features.
        n_forecast (int, optional): The number of steps to forecast. Defaults to 1.

    Returns:
        tuple: (XGBoost model, predicted values, mean squared error)
    """

    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, y_pred, mse

  def run(self, n_past=30, n_forecast=1):
    """
    Executes the entire forecasting process: data loading, preprocessing, training, and evaluation.

    Args:
        n_past (int, optional): The number of past periods to use as features. Defaults to 30.
        n_forecast (int, optional): The number of steps to forecast. Defaults to 1.
    """

    self.load_data()
    self.preprocess_data()

    X_train, y_train = self.create_lagged_dataset(n_past)
    X_test, y_test = self.create_lagged_dataset(n_past, shift=n_forecast)  # Consider using a proper train/test split

    model, y_pred, mse = self.train_forecast_evaluate(X_train, y_train, X_test)
    print(f"Mean Squared Error: {mse}")

    # Analyze forecasts (plot actual vs. predicted, calculate other metrics)
    # ...
