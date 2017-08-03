import json
import pandas as pd
import numpy as np
import datetime

class FileHelper:

    def __init__(self):
        pass

    def load_json_file(self, filename):
        with open(filename) as data_file:
            data = json.load(data_file)

        humidity_array = np.array([d["data"] for d in data])
        ts_array = np.array([datetime.datetime.fromtimestamp(d["ts"]/1000) for d in data])
        data = dict()
        for i in range(0, len(ts_array)):
            data[ts_array[i]] =  humidity_array[i]
        return data

    def convert_dataset_file(self, filename):
        data = self.load_json_file(filename)

        new_file_name = filename.replace(".json", ".csv")

        file = open(new_file_name, "w")
        file.write("timestamp, value\n")
        for k in sorted(data.keys()):
            file.write("{0}, {1}\n".format(k.isoformat(), data[k]))

        return new_file_name

    def load_csv_file(self, filename):
        def parser(x):
            return datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

        series = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        # summarize first few rows
        print(series.head())
        # line plot
        return series
