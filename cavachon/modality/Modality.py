from collections import OrderedDict
from typing import Union

import anndata
import numpy as np
import scipy
import pandas as pd
import tensorflow as tf

class Modality(anndata.AnnData):
  def __init__(
      self,
      X: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, tf.Tensor, anndata.AnnData, None],     
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
    cavachon_uns = OrderedDict((
      ('cavachon/config', cavachon_config),
    ))
    uns = kwargs.get('uns', cavachon_uns)
    uns.update(cavachon_uns)
    kwargs.pop('uns', None)

    if not isinstance(X, anndata.AnnData):
      super().__init__(X=matrix, uns=uns, *args, **kwargs)
    else:
      X.uns.update(uns)
      super().__init__(X=X)
