from __future__ import annotations
from cavachon.filter.AnnDataFilter import AnnDataFilter

import anndata
import scanpy

class FilterGenes(AnnDataFilter):
  """FilterGenes
  
  Filter for AnnData. Used as an adaptor between the 
  scanpy.pp.filter_genes() and the configs. Note that the 
  preprocessing step will be performed inplace.

  Attributes
  ----------
  name: str
      name for the filtering step.
  
  kwargs: Mapping[str, Any]
      additional parameters used for scanpy.pp.filter_genes().

  """

  def __init__(self, name, **kwargs):
    """Constructor for FilterGenes

    Parameters
    ----------
    name: str
        name for the filtering step.
  
    kwargs: Mapping[str, Any]
        additional parameters used for scanpy.pp.filter_genes().

    """
    super().__init__(name, **kwargs)

  def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
    """Perform preprocessing to provided AnnData.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData to preprocessed.

    Returns
    -------
    anndata.AnnData
        preprocessed AnnData.

    Raises
    ------
    RuntimeError
        if no variables left after the filtering.

    """
    scanpy.pp.filter_genes(adata, **self.kwargs)
    if len(adata.var) == 0:
        message = ''.join((
        'No variables left in the adata, please use a less strict '
        f'filter in {self.__class__.__name__} ({self.name}).'
        ))
        raise RuntimeError(message)

    return adata