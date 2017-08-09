import unittest
import numpy as np
from recurrent_model import RecurrentModel


class RecurrentModelTest(unittest.TestCase):

    def test_differences(self):
        rnn = RecurrentModel(10, None, None, None, None, None)
        input = [1, 2, 4, 8, 16, 15, 14, 10, -5, -4.5]
        output = rnn.calculate_differences(input)
        self.assertTrue(output==[1,2,4,8,-1,-1,-4,-15, 0.5])

    def test_normalize(self):
        rnn = RecurrentModel(10, None, None, None, None, None)
        i = [3.0, 2.0, 4.0, 16.0, 25.0, 50.0]

        # Normalization should be:   i[n]/i[0] - 1

        normalized = rnn.normalize(i)

        for x in range(len(i)):
            print("{0}:{1}, {2}".format(x, i[x], normalized[x]))
            self.assertTrue(normalized[x] == (i[x] / np.mean(i))-1)

        o = rnn.denormalize(normalized)

        # A capricious value...
        sigma = np.std(i) / 100000.0

        print("Epsilon = {0}".format(sigma))

        for x in range(len(i)):
            print("{0}: {1} -> {2} -> {3}".format(x, i[x], normalized[x], o[x]))

            self.assertTrue(abs(i[x]-o[x]) < sigma)

        # self.assertTrue(i == o)

    def test_scale(self):
        rnn = RecurrentModel( 10, None, None, None, None, None)
        i = [-34.23, -14, -9.883, 0, 3.0, 2.0, 4.0, 16.0, 25.0, 50.0]
        scaled = rnn.scale_data(i)
        self.assertTrue(max(scaled)==1)
        self.assertTrue(min(scaled)==-1)
        unscaled = rnn.unscale_data(scaled)

        # A capricious value...
        sigma = np.std(i) / 100000.0

        for x in range(len(i)):
            self.assertTrue(abs(i[x]-unscaled[x])<sigma)

    def test_split_data(self):

        #  This is a tricky test and covers probably too much stuff.  We want to show if we call rnn.split with
        #  the parameters of input_length = 3, output_length = 3, and training_record_ration of 0.5 with
        #  the following data:
        #
        #  [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ]
        #
        #  That we get the following values for train_x, train_y, test_x and test_y
        #
        #  train_x    | train_y
        #  [1, 2, 3]  | [4, 5, 6]
        #  [2, 3, 4]  | [5, 6, 7]
        #  [3, 4, 5]  | [6, 7, 8]
        #  [4, 5, 6]  | [7, 8, 9]
        #  [6, 7, 8]  | [8, 9, 10]
        #  [7, 8, 9]  | (etc.)
        #  [8, 9, 10] |
        #  [9, 10, 11]| [12, 13, 14]
        #
        #  test_x       | test_y
        #  [10, 11, 12] | [13, 14, 15]
        #  [11, 12, 13] | [14, 15, 16]
        #  [12, 13, 14] | [15, 16, 17]
        #  [13, 14, 15] | [16, 17, 18]

        rnn = RecurrentModel(10, False, False, input_length=3, output_length=2, training_record_ratio=0.5)
        i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        train_x, train_y, test_x, test_y = rnn.split_data(i)
        print("train_x: {0}".format(train_x))
        print("train_y: {0}".format(train_y))
        print("test_x:  {0}".format(test_x))
        print("test_y:  {0}".format(test_y))

        self.assertTrue(train_x[0].tolist() == [[1], [2], [3]])
        self.assertTrue(train_x[1].tolist() == [[2], [3], [4]])
        self.assertTrue(train_x[2].tolist() == [[3], [4], [5]])
        self.assertTrue(train_x[3].tolist() == [[4], [5], [6]])
        self.assertTrue(train_y[0].tolist() == [4, 5])
        self.assertTrue(train_y[1].tolist() == [5, 6])
        self.assertTrue(train_y[2].tolist() == [6, 7])
        self.assertTrue(train_y[3].tolist() == [7, 8])


        #self.assertTrue(train_x == np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[7,8,9],[8,9,10],[9,10,11]]))

