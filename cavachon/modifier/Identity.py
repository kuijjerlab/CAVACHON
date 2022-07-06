from cavachon.modifier.TensorModifier import TensorModifier
from typing import Dict

import tensorflow as tf

class Identity(TensorModifier):

  def __init__(self, name, args):
    super().__init__(name, args)
    
  def execute(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    selector = f"{self.modality}:{self.target_name}"
    target = tf.sparse.to_dense(inputs.get(selector))
    inputs[f"{selector}:{self.postfix}"] = target
    return inputs