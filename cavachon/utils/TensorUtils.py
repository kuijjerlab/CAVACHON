import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from cavachon.utils.DataFrameUtils import DataFrameUtils
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Iterable, List, Optional, Tuple

class TensorUtils:
  """TensorUtils

  Class containing multiple utility functions for tf.Tensor.

  """
  @staticmethod
  def max_n_neurons(layers: Iterable[tf.keras.layers.Layer]) -> int:
    """Get the maximum number of neurons (of tf.keras.layers.Dense) in 
    layers.

    Parameters
    ----------
    layers: Iterable[tf.keras.layers.Layer]
        layers in tf.keras.Model.
    
    Returns
    -------
    int:
        the maximum number of neurons (of tf.keras.layers.Dense) in 
        layers. Return 0 if no Dense layer in the layers.
    
    """
    current_max = 0
    for layer in layers:
      if isinstance(layer, tf.keras.layers.Dense) and current_max < layer.units:
        current_max = layer.units
    return current_max

  @staticmethod
  def remove_nan_gradients(gradients: List[tf.Tensor], clip_value=0.1) -> List[tf.Tensor]:
    """Replace nan, inf with 0 and perform gradient clipping for the 
    gradients computed by tf.GradientTape.gradient().

    Parameters
    ----------
    gradients: List[tf.Tensor]
        gradients computed by tf.GradientTape.gradient()

    clip_value: float, optional
        clip values for gradient clipping. The resulting gradient will 
        be in the range of [-clip_value, clip_value]
    
    Returns
    -------
    List[tf.Tensor]
        processed gradients.
    
    """
    for i, g in enumerate(gradients):
      gradients[i] = tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
      gradients[i] = tf.where(
          tf.math.is_inf(gradients[i]),
          tf.zeros_like(gradients[i]),
          gradients[i])
      gradients[i] = tf.where(
          gradients[i] > clip_value,
          clip_value * tf.ones_like(gradients[i]),
          gradients[i])
      gradients[i] = tf.where(
          gradients[i] < -1 * clip_value,
          -1 * clip_value * tf.ones_like(gradients[i]),
          gradients[i])
          
    return gradients

  @staticmethod
  def create_backbone_layers(
      n_layers: int = 3,
      base_n_neurons: int = 128,
      max_n_neurons: int = 1024,
      rate: int = 2,
      activation: str = 'elu',
      reverse: bool = False,
      name: Optional[str] = 'backbone_network') -> tf.keras.Model:
    """Create tf.keras.Sequential models with tf.keras.layers.Dense and
    tf.keras.layers.BatchNormalization(). The created dense layers would 
    have number of neurons 
    [`base_n_neurons`, `base_n_neurons`*`rate`, ...]. For instance, 
    with default parameters, it creates layers of:
    1. tf.keras.layers.Dense(128, activation='elu')
    2. tf.keras.layers.BatchNormalization()
    3. tf.keras.layers.Dense(256, activation='elu')
    4. tf.keras.layers.BatchNormalization()
    5. tf.keras.layers.Dense(512, activation='elu')
    6. tf.keras.layers.BatchNormalization()

    Parameters
    ----------
    n_layers: int, optional
        number of layers. Defaults to 3.

    base_n_neurons: int, optional
        base number of neurons. Defaults to 128.
    
    max_n_neurons: int, optional
        maximum number of neurons. Defaults to 1024.

    rate: int, optional
        increasing rate of number of neurons (see description of the 
        function for more details). Defaults to 2.

    activation: str, optional
        activation functions in tf.keras.layers.Dense layer. Defaults 
        to 'elu'.

    reverse: bool, optional 
        whether to decrease the number of neurons in later layers. 
        Defaults to False.
    
    name: str, optional
        name of the created tf.keras.Model. Defaults to 
        'backbone_network'.

    Returns
    -------
    tf.keras.Model
        created tf.keras.Sequential model.

    """

    layers = []
    for no_layer in range(0, n_layers):
      n_neurons = max(base_n_neurons * rate ** no_layer, max_n_neurons)
      layers.append(tf.keras.layers.Dense(n_neurons, activation=activation))
    
    if reverse:
      layers.reverse()

    return tf.keras.Sequential(layers, name=name)

  @staticmethod
  def create_tensor_from_df(
      df: pd.DataFrame,
      colnames: List[str] = []) -> Tuple[tf.Tensor, Dict[str, LabelEncoder]]:
    """Create a Tensorflow Tensor from column data (specified with 
    `colnames`) in the provided DataFrame. If the column data is a 
    categorical variable, transform it with one-hot encoded Tensor. If 
    it is a continous variable, simply transform it into a Tensorflow 
    Tensor. If no valid column data is provided, return Tensor which is 
    a zero vector.

    Args:
      df (pd.DataFrame): input DataFrame.

      colnames (List[str], optional): columns of the DataFrame that are
      used to create the Tensor. Defaults to [].

    Returns:
      Tuple[tf.Tensor, Dict[str, LabelEncoder]]: the first element is 
      the one-hot encoded Tensor. The second element is the dictionary
      of LabelEncoder used to map the categorical variable into scalar 
      representation, where the keys are the column names and the values
      are the correponding LabelEncoder. The value will be None if 
      the column data is not a continous variable.
    """
    # if no valid batch effect column is provided, use zero vector for batch effect
    encoder_dict = dict()
    n_obs, n_features = df.shape
    tensor_list = []

    for colname in colnames:
      if colname not in df.columns:
        continue
      coldata = df[colname]
      if DataFrameUtils.check_is_categorical(coldata):
        # if the column is a categorical variable, use one hot encoded tensor
        encoded_tensor, encoder = TensorUtils.create_one_hot_encoded_tensor(coldata)
        encoder_dict.setdefault(colname, encoder)
        tensor_list.append(encoded_tensor)
      else:
        # if the column is a continous variable, 
        tensor = tf.reshape(tf.convert_to_tensor(coldata, tf.float32), (-1, 1))
        encoder_dict.setdefault(colname, None)
        tensor_list.append(tensor)
    
    if len(tensor_list) == 0:
      tensor_list.append(tf.zeros((n_obs, 1)))

    return tf.concat(tensor_list, axis=1), encoder_dict

  @staticmethod
  def create_one_hot_encoded_tensor(data: pd.Series) -> Tuple[tf.Tensor, LabelEncoder]:
    """Create a one hot encoded Tensor from a Series variable.

    Parameters
    ----------
    data: pd.Series
        pd.Series of categorical variables to be transformed to one-hot
        encoded tf.Tensor.

    Returns
    -------
    Tuple[tf.Tensor, LabelEncoder]:
        the first element is the one-hot encoded Tensor, the second 
        element is the LabelEncoder used to map the categorical 
        variable into scalar representation.
    
    """
    encoder = LabelEncoder()
    encoded_array = encoder.fit_transform(data)
    n_class = len(encoder.classes_)
    encoded_tensor = tf.cast(tf.one_hot(encoded_array, n_class), tf.float32)

    return encoded_tensor, encoder

  @staticmethod
  def spmatrix_to_sparse_tensor(spmatrix: scipy.sparse.spmatrix) -> tf.SparseTensor:
    """Create a SparseTensor out of a scipy sparse matrix.

    Parameters
    ----------
    spmatrix: sparse matrix
        the provided matrix

    Returns
    -------
    tf.SparseTensor:
        the created SparseTensor.
    
    """
    coo_matrix = spmatrix.tocoo()
    indices = np.mat([coo_matrix.row, coo_matrix.col]).transpose()
    sparse_tensor = tf.SparseTensor(indices, coo_matrix.data, coo_matrix.shape)
    return tf.sparse.reorder(tf.cast(sparse_tensor, tf.float32))
  
  @staticmethod
  def split(x: tf.Tensor, batch_size: int = 128) -> List[tf.Tensor]:
    """Split the tensor on the first dimension (batch), with batch_size.

    Parameters
    ----------
    x: tf.Tensor
        input tensor.

    batch_size: int, optional
        the batch size to split. Defaults to 128

    Returns
    -------
    List[tf.Tensor]
        List of splitted tensors.
    """
    n_obs = x.shape[0]
    # if batch_size = 128, n_obs = 1000
    # split_batch = [128, 128, 128, 128, 128, 128, 128, 104]
    split_batch = [batch_size] * (n_obs // batch_size) + [n_obs % batch_size]
    
    return tf.split(x, split_batch)