from collections.abc import Callable
import anndata

class AnnDataFilter(Callable):
  """AnnDataFilter
  
  Filter for AnnData. Used as an adaptor between the scanpy 
  preprocessing methods and the configs. Note that the preprocessing
  step will be performed inplace.

  Attributes
  ----------
  name: str
      name for the filtering step.
  
  kwargs: Mapping[str, Any]
      additional parameters used for scanpy preprocessing methods.

  """
  def __init__(self, name, **kwargs):
    """Constructor for AnnDataFilter

    Parameters
    ----------
    name: str
        name for the filtering step.
  
    kwargs: Mapping[str, Any]
        additional parameters used for scanpy preprocessing methods.

    """
    self.name = name
    self.kwargs = kwargs
    self.kwargs.pop('step', None)
    self.kwargs['inplace'] = True

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

    """
    return adata