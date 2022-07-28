from cavachon.modifier.Modifier import Modifier
from typing import Dict

import tensorflow as tf

class BatchModifier(Modifier):
  def __init__(self, name, keep_original: bool = True):
    self.name = name
    self.keep_original = keep_original

  def process(
      self, 
      batch: Dict[str, tf.Tensor],
      selector: str
  ) -> tf.Tensor:
    return batch[f"{selector}"]

  def execute(
      self,
      batch: Dict[str, tf.Tensor],
      selector: str) -> Dict[str, tf.Tensor]:
    new_tensor = self.process(batch)
    batch = self.complete(new_tensor, batch, selector)
    return batch

  def complete(
      self,
      new_tensor: tf.Tensor,
      batch: Dict[str, tf.Tensor],
      selector: str,
  ) -> Dict[str, tf.Tensor]:
    if self.keep_original:
      batch.setdefault(f'{selector}/original', batch[f'{selector}'])
    else:
      batch[f'{selector}/original'] = new_tensor
    batch[f'{selector}'] = new_tensor
    return batch
