import pandas as pd
import unittest

from cavachon.utils.DataFrameUtils import DataFrameUtils

class DataFrameUtilsTestCase(unittest.TestCase):

  def test_check_is_categorical_positive_numeric(self):
    data = pd.DataFrame({"Test": [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]})
    self.assertTrue(
        DataFrameUtils.check_is_categorical(data['Test']), 
        "The result is incorrect, expect to be True.")

  def test_check_is_categorical_positive_categorical(self):
    data = pd.DataFrame({
        "Test": ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B']})
    self.assertTrue(
        DataFrameUtils.check_is_categorical(data['Test']), 
        "The result is incorrect, expect to be True.")

  def test_check_is_categorical_negative_numeric(self):
    data = pd.DataFrame({"Test": [1., 2., 3., 4., 5.]})
    self.assertFalse(
        DataFrameUtils.check_is_categorical(data['Test']), 
        "The result is incorrect, expect to be False.")
  
  def test_check_is_categorical_negative_categorical(self):
    data = pd.DataFrame({"Test": ['A', 'B', 'C', 'D', 'E']})
    self.assertFalse(
        DataFrameUtils.check_is_categorical(data['Test']), 
        "The result is incorrect, expect to be False.")

  def test_check_is_categorical_positive_threshold(self):
    data = pd.DataFrame({"Test": [1., 1., 1., 0., 0.,]})
    self.assertTrue(
        DataFrameUtils.check_is_categorical(data['Test'], 0.5), 
        "The result is incorrect, expect to be True.")
  
  def test_check_is_categorical_negative_threshold(self):
    data = pd.DataFrame({"Test": [1., 1., 1., 0., 0.,]})
    self.assertFalse(
        DataFrameUtils.check_is_categorical(data['Test'], 0.1), 
        "The result is incorrect, expect to be False.")