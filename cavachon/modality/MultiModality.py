from cavachon.modality import Modality
from collections.abc import Mapping
from typing import Union
from typing import Mapping as MappingType

import anndata
import muon

class MultiModality(muon.MuData):
  """MultiModality
  
  Data structure for (single-cell) multi-omics data. Inherit from 
  muon.MuData and is comptatible with muon and other APIs that expect 
  muon.MuData as inputs. The adata in each modality will be sorted so
  the order of the cells will be the same after initialization.

  """
  
  def __init__(
    self,
    data: Union[anndata.AnnData, Modality, MappingType[str, anndata.AnnData], muon.MuData],
    *args,
    **kwargs
  ):
    """Constructor for MultiModality

    Parameters
    ----------
    data: Union[anndata.AnnData, Modality, MappingType[str, anndata.AnnData], muon.MuData])
        same as the data parameter for muon.MuData.

    args:
        addtional parameters for initializing anndata.AnnData.

    kwargs: Mapping[str, Any]
        addtional parameters for initializing anndata.AnnData.
        
    """
    if isinstance(data, Modality):
      modality_name = data.uns.get('cavachon', {}).get('name', 'modality')
      data = dict((
          (modality_name, data),
      ))

    if isinstance(data, muon.MuData):
      muon.pp.intersect_obs(data)
      data = data.mod
    
    if issubclass(type(data), Mapping):
      index = None
      for key in data.keys():
        adata = data.get(key)
        index = adata.obs.index if index is None else index.intersection(adata.obs.index)
      for key in data.keys():
        adata = data.get(key)
        # workaround to turn a view of AnnData to real AnnData.
        adata = adata[index]
        adata.uns.setdefault('dummy', None)
        adata.uns.pop('dummy', None)
        data[key] = adata

    super().__init__(data, *args, **kwargs)