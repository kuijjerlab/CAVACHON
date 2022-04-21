import numpy as np
import pandas as pd

class DataFrameUtils:
  """DataFrameUtils
  Utility functions for Pandas Dataframe and Series
  """

  @staticmethod
  def check_is_categorical(data: pd.Series, threshold: float = 0.2) -> bool:
    """Check if the given Series variable is categorical. This function computes the
    proportion of the number of unique values, if the number of unique values is larger 
    than `threshold`, it is considered as a continuous variable.

    Args:
      data (pd.Series): data variable.

      threshold (float): threshold of proportion of the number of unique values to
      determine if one variable is categorical or not. Defaults to 0.2.

    Returns:
      bool: Ture if the series is a categorical variable, otherwise False.
    """
    n_obs = data.size
    return len(np.unique(data)) < n_obs * threshold

  