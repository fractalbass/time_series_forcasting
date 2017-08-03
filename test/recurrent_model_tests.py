import unittest
from recurrent_model import RecurrentModel


class RecurrentModelTest(unittest.TestCase):

    def test_get_target_set(self):
        rnn = RecurrentModel()

        training_set = [1,2,3]
        target_set = rnn.get_target_set(training_set)
        self.assertTrue(target_set == [2, 3])
