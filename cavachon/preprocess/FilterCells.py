from __future__ import annotations
from cavachon.preprocess.PreprocessStep import PreprocessStep
import scanpy

class FilterCells(PreprocessStep):

  def __init__(self, name, kwargs):
    super().__init__(name, kwargs)

  def execute(self, modality: Modality) -> None:
    self.kwargs['inplace'] = True
    scanpy.pp.filter_cells(modality.adata, **self.kwargs)
    return