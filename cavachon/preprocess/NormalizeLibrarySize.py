from __future__ import annotations
from cavachon.preprocess.PreprocessStep import PreprocessStep

class NormalizeLibrarySize(PreprocessStep):

  def __init__(self, name, kwargs):
    super().__init__(name, kwargs)

  def execute(self, modality: Modality) -> None:
    field = self.kwargs.get('field', None)
    if field is not None and field in modality.adata.obs.columns:
      libsize = modality.adata.obs[field].values
    else:
      libsize = modality.adata.X.sum(axis=1)
      modality.adata.obs['libsize'] = libsize
    modality.adata.X  = modality.adata.X / libsize
    
    return