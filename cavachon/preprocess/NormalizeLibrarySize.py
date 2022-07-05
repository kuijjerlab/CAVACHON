from __future__ import annotations
from cavachon.preprocess.PreprocessStep import PreprocessStep

import numpy as np

class NormalizeLibrarySize(PreprocessStep):

  def __init__(self, name, kwargs):
    super().__init__(name, kwargs)

  def execute(self, modality: Modality) -> None:
    field = self.kwargs.get('field', None)
    if field is not None and field in modality.adata.obs.columns:
      libsize = np.reshape(modality.adata.obs[field].values, (-1, 1))
    else:
      libsize = modality.adata.X.sum(axis=1)
      modality.adata.obs['libsize'] = libsize
    
    coo_matrix = modality.adata.X.tocoo()
    scale_fn = np.vectorize(lambda x: libsize[x, 0])
    scale_factor = scale_fn(coo_matrix.row)
    min_nonzero = np.min(modality.adata.X[modality.adata.X != 0])
    normalized_x = modality.adata.X[modality.adata.X >= min_nonzero] / scale_factor
    modality.adata.X[modality.adata.X >= min_nonzero] = normalized_x
    
    return