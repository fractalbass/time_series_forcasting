from matplotlib import pyplot
from recurrent_model import RecurrentModel
from datetime import datetime
from file_helper import FileHelper
import pandas as pd

class HumidityModel:

    def __init__(self):
        pass

    def run(self):
        print("Starting...")
        # Load the note file
        file_helper = FileHelper()
        new_file = file_helper.convert_dataset_file("./data/humidity_training_and_testing.json")
        # new_file = "./data/sinewave.csv"
        print("New file {0} has been created.".format(new_file))

        rnn = RecurrentModel(epochs=100, use_differences=False, use_normalization=False, input_length=40,
                             output_length=10, training_record_ratio=0.5)

        rnn.prepare_data(new_file)

        # Train the model
        rnn.train_network()

        # Save the model
        rnn.save_model("{0}_model_{1}".format(new_file, datetime.now().timestamp()))

        # Display the graph
        rnn.display_results()

        print("Done.")

if __name__ == "__main__":
    bn = HumidityModel()
    bn.run()
