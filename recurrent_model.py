import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

class RecurrentModel:

    def __init__(self, epochs, use_differences, use_normalization, input_length, output_length, training_record_ratio):
        self.epochs = epochs
        self.use_differences = use_differences
        self.use_normalization = use_normalization
        self.input_length = input_length
        self.output_length = output_length
        self.training_record_ratio = training_record_ratio
        self.batch_size = 32

    def load_data(self, filename):
        f = open(filename, 'rb').read()
        raw_data = f.decode().split('\n')
        values = []

        for x in raw_data:
            try:
                values.append(float(x))
            except Exception:
                print("Failed to convert {0} to a float value.".format(x))

        return values

    def calculate_differences(self, data):
        values = []
        for d in range(1,len(data)):
            values.append(data[d]-data[d-1])
        return values

    def normalize(self, data):
        self.average = np.mean(data)
        normalized_data = [((float(p) / self.average) - 1) for p in data]
        return normalized_data

    def denormalize(self, data):
        denormalized = [((1+d)*self.average) for d in data]
        return denormalized

    def scale_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = self.scaler.fit(data)
        result = self.scaler.transform(data)
        return result

    def unscale_data(self, data):
        return self.scaler.inverse_transform(data)

    def split_data(self, data):

        x_result = []

        for index in range(len(data) - (self.output_length + self.input_length) + 1):
            x_result.append(data[index: index + self.input_length])

        y_result = []

        for index in range(self.input_length, len(data) - self.output_length + 1):
            y_result.append(data[index: index + self.output_length])

        x_result = np.array(x_result)
        y_result = np.array(y_result)
        row = round(self.training_record_ratio * x_result.shape[0])

        inputs =  x_result[:int(row), :]
        outputs = y_result[:int(row), :]
        # The x-train training length is everything up to the split
        # this will be the input to the network for this record in the sequence.
        x_train = x_result[:row]

        # The y_train length is the end of the set, where the count of elements is the "output_length"
        y_train = y_result[:row]

        # Same as above, but for the testing sequence
        x_test = x_result[int(row):]

        # Same as above, but for the testing sequence
        y_test = y_result[int(row):]

        # Here we put everything into a shape that the network can act on.

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # <- may be output_length for last val?
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_train, y_train, x_test, y_test

    def prepare_data(self, filename):
        data = self.load_data(filename)

        if self.use_differences:
            data = self.calculate_differences(data)

        if self.use_normalization:
            data = self.normalize(data)

        data = self.scale_data(data)

        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data(data)

    def build_model(self):

        hidden_neurons = 300
        model = Sequential()

        model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))

        model.add(Dense(self.output_length))
        model.add(Activation("linear"))
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        model.summary()
        self.model = model

    def train_network(self):
        global_start_time = time.time()

        print('> Data Loaded. Compiling...')

        self.build_model()

        #model_input = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))

        self.model.fit(
            self.X_train,
            self.y_train,
            nb_epoch=self.epochs,
            validation_split=0.05)

        print('Training duration (s) : ', time.time() - global_start_time)

    def save_model(self, filename):
        full_filename = "{0}.h5".format(filename)
        print("Saving file: {0}".format(full_filename))
        self.model.save(full_filename)
        print("File saved.")

    def load_model(self, filename):
        print("Loading model: {0}".format(filename))
        self.model = load_model(filename)
        print("File loaded.")

# ======== This file is getting too long.

    def display_results(self):
        self.predict_point_by_point()
        self.plot_results()

    def predict_point_by_point(self):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        self.predicted = self.model.predict(self.X_test)

    def plot_results(self):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        # ax.plot(self.y_test)

        net_predicted = []
        actual = []

        for index in range(0, len(self.X_test), self.output_length):
            # l = lines.Line2D([index,index],[-1,1])
            # ax.add_line(l)

            for offset in range(len(self.y_test[index])):
                x = index + offset
                net_predicted.append(self.predicted[index][offset])
                actual.append(self.y_test[index][offset])
                print("{0}, {1}, {2}".format(x, self.predicted[index][offset], self.y_test[index][offset]))

        ax.plot(net_predicted, label="predicted")
        ax.plot(actual, label="actual")
        plt.legend()
        plt.show()

    def predict_sequence_full(self, model, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        return predicted

    def predict_sequences_multiple(self, model, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs



    def plot_results_multiple(predicted_data, true_data, prediction_len):

        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()



