from collections.abc import Callable
import anndata

class AnnDataFilter(Callable):

  def __init__(self, name, **kwargs):
    self.name = name
    self.kwargs = kwargs
    self.kwargs.pop('step', None)
    self.kwargs['inplace'] = True

  def __call__(self, adata: anndata) -> anndata.AnnData:
    return adata