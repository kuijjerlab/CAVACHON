import numpy as np
import pandas as pd

class DataFrameUtils:
  """DataFrameUtils

  Class containing multiple utility functions for pd.DataFrame or 
  pd.Series.
  
  """

  @staticmethod
  def check_is_categorical(data: pd.Series, threshold: float = 0.2) -> bool:
    """Check if the given Series variable is categorical. This function
    computes the proportion of the number of unique values, if the
    number of unique values is larger than `threshold`, it is considered
    as a continuous variable.

    Parameters
    ----------
    data: pd.Series
        pd.Series needs to be check.

    threshold: float
        threshold of proportion of the number of unique values to 
        determine if one variable is categorical or not. Defaults to 
        0.2.

    Returns
    -------
    bool:
        Ture if the series is a categorical variable, otherwise False.
    """
    n_obs = data.size
    return len(np.unique(data)) < n_obs * threshold

  