from cavachon.modifier.TensorModifier import TensorModifier
from typing import Dict

import tensorflow as tf

class Binarize(TensorModifier):

  def __init__(self, name, args):
    super().__init__(name, args)
    self.threshold = args.get('threshold', 1.0)
    
  def execute(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    selector = f"{self.modality}:{self.target_name}"
    target = tf.sparse.to_dense(inputs.get(selector))
    target = tf.where(target >= self.threshold, tf.ones_like(target), target)
    target = tf.where(target < self.threshold, tf.zeros_like(target), target)
    inputs[f"{selector}:{self.postfix}"] = tf.sparse.from_dense(target)
    return inputs