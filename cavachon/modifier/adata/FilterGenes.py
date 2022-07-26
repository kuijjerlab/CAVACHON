from __future__ import annotations
from cavachon.modifier.adata.AnnDataFilter import AnnDataFilter

import anndata
import scanpy

class FilterGenes(AnnDataFilter):

  def __init__(self, name, args):
    super().__init__(name, args)

  def execute(self, adata: anndata.AnnData) -> anndata.AnnData:
    scanpy.pp.filter_genes(adata, *self.args, **self.kwargs)
    return adata