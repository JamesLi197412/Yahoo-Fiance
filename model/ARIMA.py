import statsmodels.api as sm
import matplotlib.pyplot as plt

# ARIMA models are effective for capturing linear trends and seasonal patterns

class ARIMA_Model:
      def __init__(self, data, order):
        self.data = data
        self.order = order
        self.train = None

      def split_data(self, ratio):
          X = self.data.values
          train_size = int(len(X) * ratio)

          train, test = X[:train_size], X[train_size:len(X)]
          return train, test, train_size

      def build_model(self,history):
        """
        Builds and fits an ARIMA model to the data.
        """
        model = sm.tsa.arima.ARIMA(history, order=self.order)
        model_fit = model.fit()

        # Make the prediction
        output = model_fit.forecast()
        y_pred = output[0]
        return y_pred


      def run(self):
          train,test,train_size = self.split_data(0.8)
          history = [x for x in train]
          predictions = list()
          for t in range(len(test)):
              yhat = self.build_model(history)
              predictions.append(yhat)
              obs = test[t]
              history.append(obs)

          self.visulations(train_size,test,predictions)
          return predictions

      def visulations(self,train_size,test,prediction):
          plt.figure(figsize=(12, 6), dpi=100)
          plt.plot(self.data.iloc[train_size:, :].index,test, label='Real')
          plt.plot(self.data.iloc[train_size:, :].index,prediction, color='red', label='Predicted')
          plt.title('ARIMA Predictions vs Actual Values')
          plt.xlabel('Date')
          plt.ylabel('Stock Price')
          plt.legend()
          plt.show()
