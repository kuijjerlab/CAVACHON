from __future__ import annotations
from cavachon.filter.AnnDataFilter import AnnDataFilter

import anndata
import scanpy

class FilterGenes(AnnDataFilter):

  def __init__(self, name, **kwargs):
    super().__init__(name, **kwargs)

  def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
    scanpy.pp.filter_genes(adata, **self.kwargs)
    return adata