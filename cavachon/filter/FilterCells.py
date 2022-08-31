from cavachon.filter.AnnDataFilter import AnnDataFilter

import anndata
import scanpy

class FilterCells(AnnDataFilter):

  def __init__(self, name, **kwargs):
    super().__init__(name, **kwargs)

  def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
    scanpy.pp.filter_cells(adata, **self.kwargs)
    return adata
