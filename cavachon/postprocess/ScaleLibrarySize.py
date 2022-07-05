from cavachon.postprocess.PostprocessStep import PostprocessStep
from typing import Dict

import tensorflow as tf

class ScaleLibrarySize(PostprocessStep):

  def __init__(self, name: str, modality_name: str):
    super().__init__(name, modality_name)

  def execute(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    matrix = tf.sparse.to_dense(inputs[f'{self.modality_name}:matrix'])
    libsize = inputs[f'{self.modality_name}:libsize']
    inputs[f'{self.modality_name}:matrix'] = tf.sparse.from_dense(matrix * libsize)
    return inputs