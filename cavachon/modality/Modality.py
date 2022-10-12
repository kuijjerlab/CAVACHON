from cavachon.environment.Constants import Constants
from collections import OrderedDict
from typing import Collection, Optional, Union

import anndata
import numpy as np
import scipy
import pandas as pd
import tensorflow as tf

class Modality(anndata.AnnData):
  """Modality
  
  Data structure for (single-cell) single-omics data. Inherit from 
  anndata.AnnData and is comptatible with scanpy and other APIs that
  expect andata.AnnData as inputs.

  Attributes
  ----------
  uns: Any
      same as the unstructure annotations from anndata.AnnData, but
      with additional config annotations specifically for CAVACHON, 
      which is stored as a dictionary in uns['cavcachon/config']. The
      additional information includes the 'name', 'modality_type', 
      'distribution' and 'batch_effect_colnames' for the modality.
  """

  def __init__(
      self,
      X: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, tf.Tensor, anndata.AnnData, None],     
      name: str,
      modality_type: str,
      distribution_name: Optional[str] = None,
      batch_effect_colnames: Optional[Collection[str]] = None,
      *args,
      **kwargs):
    """Constructor for Modality

    Parameters
    ----------
    X: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, tf.Tensor, anndata.AnnData, None]):
        same as the X parameter for anndata.AnnData, but support 
        tf.Tensor as inputs (will be transformed to numpy array).
    
    name: str
        the name for the modality.

    modality_type: str
        the modality type (see also Constants.SUPPORTED_MODALITY_TYPES)
      
    distribution_name: str, optional: 
        distribution for the modality. Defaults to None.

    batch_effect_colnames: Collection[str], optional
        the column names in the obs DataFrame that will be treated as
        batch effect. Defaults to None.
    
    args:
        addtional parameters for initializing anndata.AnnData.

    kwargs: Mapping[str, Any]
        addtional parameters for initializing anndata.AnnData.
    """

    if isinstance(X, tf.Tensor):   
      matrix = np.array(X)
    else:
      matrix = X

    if distribution_name is None:
      distribution_name = Constants.DEFAULT_MODALITY_DISTRIBUTION.get(modality_type.lower())

    cavachon_config = dict((
      ('name', name),
      ('modality_type', modality_type),
      ('distribution', distribution_name),
      ('batch_effect_colnames', batch_effect_colnames)
    ))
    cavachon_uns = OrderedDict((
      ('cavachon', cavachon_config),
    ))
    uns = kwargs.get('uns', cavachon_uns)
    uns.update(cavachon_uns)
    kwargs.pop('uns', None)

    if not isinstance(X, anndata.AnnData):
      super().__init__(X=matrix, uns=uns, *args, **kwargs)
    else:
      X.uns.update(uns)
      super().__init__(X=X)
