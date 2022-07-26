from cavachon.modifier.Modifier import Modifier

import anndata

class AnnDataFilter(Modifier):

  def __init__(self, name, *args, **kwargs):
    self.name = name
    self.args = args
    self.kwargs = kwargs
    self.kwargs.pop('name', None)
    self.kwargs.pop('func', None)
    self.kwargs['inplace'] = True

  def execute(self, adata: anndata) -> anndata.AnnData:
    return adata