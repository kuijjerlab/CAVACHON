from collections.abc import Callable
import anndata

class AnnDataFilter(Callable):

  def __init__(self, name, *args, **kwargs):
    self.name = name
    self.args = args
    self.kwargs = kwargs
    self.kwargs.pop('name', None)
    self.kwargs.pop('func', None)
    self.kwargs['inplace'] = True

  def __call__(self, adata: anndata) -> anndata.AnnData:
    return adata