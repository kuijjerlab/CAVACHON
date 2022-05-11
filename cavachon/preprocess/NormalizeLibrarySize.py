import tensorflow as tf

from cavachon.preprocess.PreprocessBatch import PreprocessBatch
from typing import Dict

class NormalizeLibrarySize(PreprocessBatch):

  @staticmethod
  def execute(batch: Dict[str: tf.Tensor], modality: str) -> Dict[str, tf.Tensor]:
    X_name = f'{modality}:matrix'
    libsize_name = f'{modality}:libsize'
    X = batch.get(X_name, None)
    libsize = batch.get(libsize_name, None)
    batch.update({X_name: X / libsize})
    return batch