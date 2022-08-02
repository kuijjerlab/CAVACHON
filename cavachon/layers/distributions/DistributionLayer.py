from abc import ABC, abstractstaticmethod
import tensorflow as tf
import tensorflow_probability as tfp

class DistributionLayer(ABC):
  @abstractstaticmethod
  def dist(*args, **kwargs) -> tfp.distributions.Distribution:
    return

  @abstractstaticmethod
  def prob(*args, **kwargs) -> tf.Tensor:
    return

  @abstractstaticmethod
  def sample(*args, **kwargs) -> tf.Tensor:
    return