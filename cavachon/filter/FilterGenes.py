from __future__ import annotations
from cavachon.filter.FilterStep import FilterStep
import scanpy

class FilterGenes(FilterStep):

  def __init__(self, name, args):
    super().__init__(name, args)

  def execute(self, modality: Modality) -> None:
    self.args['inplace'] = True
    scanpy.pp.filter_genes(modality.adata, **self.args)
    modality.n_obs = modality.adata.n_obs
    modality.n_vars = modality.adata.n_vars
    return