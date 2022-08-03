from cavachon.utils.TensorUtils import TensorUtils
from typing import Union

import anndata
import numpy as np
import scipy
import pandas as pd
import tensorflow as tf

class Modality(anndata.AnnData):
  def __init__(
      self,
      X: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, tf.Tensor, None],     
      name: str,
      modality_type: str,
      order: int,
      n_layers: int,
      n_clusters: int,
      n_latent_dims: int, 
      *args,
      **kwargs):

    if isinstance(X, tf.Tensor):   
      matrix = np.array(X)
    else:
      matrix = X
    
    cavachon_config = dict((
      ('name', name),
      ('modality_type', modality_type),
      ('order', order),
      ('n_layers', n_layers),
      ('n_clusters', n_clusters),
      ('n_latent_dims', n_latent_dims),
    ))
    cavachon_uns = dict((
      ('cavachon/config', cavachon_config),
    ))
    uns = kwargs.get('uns', cavachon_uns)
    uns.update(cavachon_uns)
    kwargs.pop('uns', None)

    super().__init__(X=matrix, uns=uns, *args, **kwargs)

  def __repr__(self) -> str:
    message = super().__repr__()
    message += '\n    tensors:'
    for i, key in enumerate(self.tensor.keys()):
      if i != 0:
        message += ","
      message += f" '{key}'"
    message += '\n'
    return message

  def _init_as_actual(self, *args, **kwargs) -> None:
    super()._init_as_actual(*args, **kwargs)
    self._reset_tensor()
   
  def _reset_tensor(self) -> None:
    if isinstance(self.X, (np.ndarray, pd.DataFrame)):   
      matrix = np.array(self.X)
      tensor = tf.convert_to_tensor(matrix)
    elif isinstance(self.X, scipy.sparse.spmatrix):
      matrix = scipy.sparse.coo_matrix(self.X)
      tensor = TensorUtils.spmatrix_to_sparse_tensor(matrix)
    self.tensor = dict((
      ('X', tensor),
    ))

  @property
  def X(self):
    return super().X

  @X.setter
  def X(self, value) -> None:
    super(Modality, type(self)).X.fset(self, value)
    self._reset_tensor()