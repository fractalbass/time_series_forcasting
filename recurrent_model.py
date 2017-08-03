from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RecurrentModel:

    def __init__(self):
        self.model = Sequential()
        self.scaler = None

    def train_network(self, training_set, batch_size, nb_epoch, neurons):
            full_loss = []
            training_set = self.scale_data(training_set)[:-1]
            target_set = self.get_target_set(training_set)
            training_set = np.reshape(training_set, (training_set.shape[0], 1, 0))
            self.model.add(LSTM(neurons, batch_input_shape=(batch_size, 1, 1), stateful=True))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            hist = self.model.fit(training_set, target_set, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)

            return hist

    def get_target_set(self, training_set):
        target_set = []

        for c in range(0, len(training_set) - 1):
            target_set.append(training_set[c + 1])
            print("{0}:{1}".format(training_set[c], target_set[c]))

        return target_set

    def predict(self, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = self.model.predict(X, batch_size=batch_size)
        return self.unscale_data(yhat[0, 0])

    def scale_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = self.scaler.fit(data)
        # transform train
        scaled_data = self.scaler.transform(data)
        return scaled_data

    def unscale_data(self, data):
        return self.unscale_data(data)
