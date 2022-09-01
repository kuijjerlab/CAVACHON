#%%
from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import List, Mapping
import tensorflow as tf

class Preprocessor(tf.keras.Model):
  def __init__(self, inputs, outputs, name, matrix_key):
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.matrix_key = matrix_key

  @classmethod
  def make(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str],
      n_vars: Mapping[str, int],
      n_dims: int = 1024,
      name: str = 'preprocessor',
      **kwargs) -> tf.keras.Model:
    
    inputs = dict()
    outputs = dict()
    processed_matrix = list()

    for modality_name in modality_names:
      distribution_name = distribution_names.get(modality_name)
      modality_key = (modality_name, Constants.TENSOR_NAME_X)
      libsize_key = (modality_name, Constants.TENSOR_NAME_LIBSIZE)
      modality_input = tf.keras.Input(
          shape=(n_vars.get(modality_name), ),
          name=modality_name)
      inputs.setdefault(modality_key, modality_input)
      
      modifiers_class = ReflectionHandler.get_class_by_name(
          distribution_name,
          'modules/preprocessors/modifiers')
      modifiers = modifiers_class(modality_name=modality_name)
      modifiers_outputs = modifiers(inputs)

      if libsize_key in modifiers_outputs:
        outputs.setdefault(libsize_key, modifiers_outputs.get(libsize_key))

      transform_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(n_dims)
      ], name=f'{name}_{modality_name}')
      processed_matrix.append(transform_layer(modifiers_outputs.get(modality_key)))

      outputs.update()
    
    matrix_key = (Constants.TENSOR_NAME_X, )
    outputs.setdefault(matrix_key, tf.concat(processed_matrix, axis=-1))

    return cls(inputs=inputs, outputs=outputs, name=name, matrix_key=matrix_key)

  def train_step(self, data):
    raise NotImplementedError()