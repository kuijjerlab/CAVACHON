from cavachon.modifier.TensorModifier import TensorModifier
from typing import Dict

import tensorflow as tf

class NormalizeLibrarySize(TensorModifier):

  def __init__(self, name, args):
    super().__init__(name, args)
    self.libsize_name = args.get('libsize_name', None)

  def execute(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    selector_target = f"{self.modality}:{self.target_name}"
    target = tf.sparse.to_dense(inputs.get(selector_target))
    
    selector_libsize = f"{self.modality}:{self.libsize_name}"
    libsize = inputs.get(selector_libsize, tf.reduce_sum(target, axis=1))
    
    inputs[f"{selector_target}:{self.postfix}"] = tf.sparse.from_dense(target / libsize)

    return inputs