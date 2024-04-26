import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler


# LSTM models tends to perform well on time series data with complex patterns and long-term dependencies

class LSTM_Model:
      def __init__(self, data, order):
        self.data = data
        self.order = order
        self.train = None

      def split_data(self, ratio,df):

          train_size = int(len(df) * ratio)

          train, test = df[:train_size], df[train_size:len(df)]
          return train, test, train_size

      def data_process(self):
          scaler = MinMaxScaler()
          df = scaler.fit_transform(np.array(self.data).reshape(-1,1))
          return df

      def create_dataset(self,time_step = 1):
          dataX, dataY = [], []
          for i in range(len(self.data) - time_step - 1):
              a = self.data[i:(i+time_step), 0]
              dataX.append(a)
              dataY.append(self.data[i + time_step,0])
          return np.array(dataX), np.array(dataY)


      def build_model(self,X_train):
          regressor = Sequential()
          # First LSTM layer with Dropout regularisation
          regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
          regressor.add(Dropout(0.3))

          regressor.add(LSTM(units=80, return_sequences=True))
          regressor.add(Dropout(0.1))

          regressor.add(LSTM(units=50, return_sequences=True))
          regressor.add(Dropout(0.2))

          regressor.add(LSTM(units=30))
          regressor.add(Dropout(0.3))

          regressor.add(Dense(units=1))

          regressor.compile(optimizer='adam', loss='mean_squared_error')
          return regressor


      def run(self, scaler,time_step = 100):
          train,test,train_size = self.split_data(0.8)
          X_train, Y_train = self.create_dataset(train,time_step)
          X_test, Y_test = self.create_dataset(test, time_step)
          model = self.build_model(X_train)
          model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)

          train_predict = model.predict(X_train)
          test_predict = model.predict(X_test)

          # transform back to original form
          train_predict = scaler.inverse_transform(train_predict)
          test_predict = scaler.inverse_transform(test_predict)
          return train_predict,test_predict

      def visulations(self,train_size,test,prediction):
          plt.figure(figsize=(12, 6), dpi=100)
          plt.plot(self.data.iloc[train_size:, :].index,test, label='Real')
          plt.plot(self.data.iloc[train_size:, :].index,prediction, color='red', label='Predicted')
          plt.title('ARIMA Predictions vs Actual Values')
          plt.xlabel('Date')
          plt.ylabel('Stock Price')
          plt.legend()
          plt.show()




class StockForecaster:
  """
  XGBoost model for stock price forecasting.
  """

  def __init__(self,df,features,target_col,parms,drop_cols):
    self.data = df
    self.features = features
    self.target = target_col
    self.parms = parms
    self.model = None
    self.drop_cols = drop_cols

  def data_preprocessing(self,split_ratio, step, features):
      # Split the data into train & test
      train_df, test_df, train_size = split_data(self.data, split_ratio)

      # Generate Dataset
      X_train, y_train = create_dataset(train_df, step, features[0])
      X_test, y_test = create_dataset(test_df, step, features[0])

      return X_train, y_train, X_test, y_test, train_size


  def data_split(self,test_size, valid_size):
    self.data = self.data.fillna(0)
    self.data.index = self.data.Date

    test_split_idx = int(self.data.shape[0] * (1-test_size))
    valid_split_idx = int(self.data.shape[0] * (1 - (valid_size + test_size)))

    train_df = self.data.iloc[:valid_split_idx].copy()
    valid_df = self.data.iloc[valid_split_idx + 1:test_split_idx].copy()
    test_df = self.data.iloc[test_split_idx+1 :].copy()

    train_df = train_df.drop(self.drop_cols,1)
    test_df = test_df.drop(self.drop_cols, 1)
    valid_df = valid_df.drop(self.drop_cols, 1)


    y_train = train_df[self.target].copy()
    X_train = train_df.drop([self.target],1)

    y_valid = valid_df[self.target].copy()
    X_valid = valid_df.drop([self.target], 1)

    y_test = test_df[self.target].copy()
    X_test = test_df.drop([self.target], 1)
    self.test_split_idx = test_split_idx
    return X_train, y_train, X_test, y_test, X_valid, y_valid

  def fine_tune(self,X_train,y_train,X_valid,y_valid,X_test,y_test):
    eval_set = [(X_train,y_train),(X_valid,y_valid)]
    model = xgb.XGBRegressor(eval_set = eval_set, objective = 'reg:squarederror', verbose = False)
    clf = GridSearchCV(model, self.parms)
    clf.fit(X_train,y_train)
    print(f"Best params:{clf.best_params_}")
    print(f"Best Validation Score = {clf.best_score_}")

    model = xgb.XGBRegressor(**clf.best_params_, objective = 'reg:squarederror')
    model.fit(X_train,y_train, eval_set = eval_set, verbose = False)

    plot_importance(model)


    y_pred = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred[:5]}')

    print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

    predicted_prices = self.data.loc[self.test_split_idx +1:].copy()
    predicted_prices[self.target] = y_pred
    return predicted_prices

  def model_run(self):
    X_train, y_train, X_test, y_test, X_valid, y_valid = self.data_split(0.15, 0.15)
    prediction = self.fine_tune(X_train, y_train, X_valid, y_valid, X_test, y_test)
    return prediction

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




def create_dataset(df,lookback, attribute):
    df = df[[attribute]].values
    X, y = [], []
    for i in range(len(df) - lookback):
        feature = df[i:i+lookback]
        target = df[i+1: i+lookback+1]
        X.append(feature)
        y.append(target)

    #return X,y
    return torch.Tensor(X),torch.Tensor(y)

