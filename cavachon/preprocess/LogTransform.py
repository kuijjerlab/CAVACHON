from __future__ import annotations
from cavachon.preprocess.PreprocessStep import PreprocessStep

import numpy as np

class LogTransform(PreprocessStep):

  def __init__(self, name, kwargs):
    super().__init__(name, kwargs)
    
  def execute(self, modality: Modality) -> None:
    modality.adata.X[modality.adata.X >= 1e-7] = np.log(modality.adata.X[modality.adata.X >= 1e-7] + 1)
    return