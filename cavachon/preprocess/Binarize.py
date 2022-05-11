import tensorflow as tf

from cavachon.preprocess.PreprocessBatch import PreprocessBatch
from typing import Dict

class Binarize(PreprocessBatch):

  @staticmethod
  def execute(batch: Dict[str: tf.Tensor], modality: str) -> Dict[str, tf.Tensor]:
    X_name = f'{modality}:matrix'
    X = batch.get(X_name, None)
    X = tf.where(X > 1, tf.ones_like(X), X)
    batch.update({X_name: X})
    return batch