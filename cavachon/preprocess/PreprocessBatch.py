import tensorflow as tf

from typing import Dict
from abc import ABC, abstractstaticmethod

class PreprocessBatch(ABC):

  @abstractstaticmethod
  def execute(batch: Dict[str: tf.Tensor], modality: str) -> Dict[str, tf.Tensor]:
    pass