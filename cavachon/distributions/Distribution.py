from abc import ABC, abstractclassmethod
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class Distribution(ABC):
  
  @abstractclassmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs) -> tfp.distributions.Distribution:
    return