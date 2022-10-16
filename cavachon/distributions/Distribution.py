from abc import ABC, abstractclassmethod
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class Distribution(ABC):
  """Abstract class of Distribution, which defines an interface of 
  from_parameterizer_output().

  """
  @abstractclassmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs) -> tfp.distributions.Distribution:
    """Create Tensorflow Probability Distribution from the outputs of
    parameterizers.

    Parameters
    ----------
    params: Union[tf.Tensor, Mapping[str, tf.Tensor]]
        Parameters for the distribution created by parameterizers. 
        Alternatively, a mapping of tf.Tensor with parameter name as 
        keys can be provided.

    Returns
    -------
    tfp.distributions.Distribution
        Created Tensorflow Probability Distribution.
    
    See Also
    --------
    Parameterizer: the modules used to create parameters for 
    distributions.
    
    """
    return