from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Any, List, Mapping
import tensorflow as tf

class Preprocessor(tf.keras.Model):
  """Preprocessor

  Preprocessor module used in components. The batch data from 
  tf.data.Dataset will be first put into this preprocessor, which will
  apply corresponding modifiers based on the specified distribtion
  in each modality. Afterwards, it will concatenate the processed
  data into one single tf.Tensor stored in the outputs Mapping. This 
  base module is implemented using Tensorflow functional API.

  Attributes
  ----------
  matrix_key: Any
      the key used to get access to the processed tf.Tensor.

  """
  def __init__(
      self,
      inputs: Mapping[Any, tf.keras.Input],
      outputs: Mapping[Any, tf.Tensor],
      name: str = 'preprocessor',
      matrix_key: Any = (Constants.TENSOR_NAME_X, )):
    """Constructor for Preprocessor. Should not be called directly most
    of the time. Please use make() to create the model.

    Arguments
    ---------
    inputs: Mapping[Any, tf.keras.Input]): 
        inputs for building tf.keras.Model using Tensorflow functional 
        API. Expect to have key (modality_name, Constants.TENSOR_NAME_X)
        by defaults, and optionally expect 
        (modality_name, Constants.LIBSIZE) for modality that needs to 
        be scaling by library size.
    
    outputs: Mapping[Any, tf.Tensor]):
        outputs for building tf.keras.Model using Tensorflow functional
        API. Have key (self.matrix_key)
    
    name: str, optional:
        Name for the tensorflow model. Defaults to 'preprocessor'.

    matrix_key: Any, optional 
        key to access the processed tf.Tensor. Defaults to 
        (Constants.TENSOR_NAME_X, ).

    """
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
    """Make the tf.keras.Model using the functional API of Tensorflow.
        
    Parameters
    ----------
    modality_names: List[str]
        names for the modality needs to be processed.
    
    distribution_names: Mapping[str, str]
        distribution for each modality. The key should be the names for
        each modality and the values are the names for corresponding
        distributions.

    n_vars: Mapping[str, int]
        number of variables for the inputs Tensors. The key should be 
        the names for each modality and the values number of feature
        in the last dimension of the modality Tensor.

    n_dims: int, optional
        number of dimension to reduce to before concatenation. Defaults
        to 1024.

    name: str, optional
        Name for the tensorflow model. Defaults to 'preprocessor'.

    Returns
    -------
    tf.keras.Model
        Created model using Tensorflow functional API.
    
    """
    
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