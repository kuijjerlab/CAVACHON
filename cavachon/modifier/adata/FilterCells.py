from cavachon.modifier.adata.AnnDataFilter import AnnDataFilter

import anndata
import scanpy

class FilterCells(AnnDataFilter):

  def __init__(self, name, *args, **kwargs):
    super().__init__(name, *args, **kwargs)

  def execute(self, adata: anndata.AnnData) -> anndata.AnnData:
    scanpy.pp.filter_cells(adata, *self.args, **self.kwargs)
    return adata

#%%
