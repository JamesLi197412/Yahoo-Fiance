import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler


# LSTM models tends to perform well on time series data with complex patterns and long-term dependencies

class LSTM_Model:
      def __init__(self, data):
        self.data = data
        self.train = None

      def split_data(self, ratio, df):
          train_size = int(len(df) * ratio)
          train, test = df[:train_size,:], df[train_size:len(df),:1]
          return train, test, train_size

      def data_process(self):
          scaler = MinMaxScaler()
          self.scaler = scaler
          df = scaler.fit_transform(np.array(self.data).reshape(-1,1))
          print(df.shape)
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
          print(regressor.summary())
          return regressor

      def rmse_evaulation(train,predict):
         return (math.sqrt(mean_squared_error(train,predict)))

      def run(self, scaler,time_step = 100):
          df = self.data_process()
          train,test,train_size = self.split_data(0.8,df)
          X_train, Y_train = self.create_dataset(train,time_step)
          X_test, Y_test = self.create_dataset(test, time_step)
          model = self.build_model(X_train)
          model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)

          train_predict = model.predict(X_train)
          test_predict = model.predict(X_test)

          # transform back to original form
          train_predict = self.scaler.inverse_transform(train_predict)
          test_predict = self.scaler.inverse_transform(test_predict)
          return train_predict,test_predict

      def visulations(self,df,look_back,train_predict,test_predict):
          trainPlot = np.empty_like(self.data)
          trainPlot[:,:] = np.nan
          trainPlot[look_back : len(train_predict) + look_back, :] = train_predict
          testPlot = np.empty_like(self.data)
          testPlot[:,:] = np.nan
          testPlot[len(train_predict) + (look_back) * 2 + 1 :len(self.data) - 1, :] = test_predict

          plt.plot(self.scaler.inverse_transform(self.data))
          plt.plot(trainPlot)
          plt.plot(testPlot)
          plt.show()






