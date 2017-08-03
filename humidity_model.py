from matplotlib import pyplot
# from recurrent_model import RecurrentModel
from file_helper import FileHelper
import pandas as pd

class HumidityModel:

    def __init__(self):
        pass

    def run(self):
        print("Starting...")
        # Load the note file
        fu = FileHelper()
        new_file = fu.convert_dataset_file("./data/office_humidity.json")
        print("New file {0} has been created.".format(new_file))

        # Now, load the new file into a something
        full_data_series = fu.load_csv_file(new_file)

        full_data_series.plot()
        pyplot.show()

        # rnn = RecurrentModel()

        # Train the network
        # loss = rnn.train_network(y, 1, 100, 10)

        # Display the graph
        # pyplot.plot(loss)
        # pyplot.show()

        # Predict a bass line

        # Play the result
        print("Done.")

if __name__ == "__main__":
    bn = HumidityModel()
    bn.run()
