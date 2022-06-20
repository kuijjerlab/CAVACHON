from types import ClassMethodDescriptorType
import pandas as pd

from anndata import AnnData
from collections import OrderedDict
from cavachon.environment.Constants import Constants
from cavachon.modality import Modality
from typing import Any, Dict, Optional

class ModalityOrderedMap:
  def __init__(self, *args, **kwargs) -> None:
    self.data: OrderedDict[str, Modality] = OrderedDict(*args, **kwargs)
    self.reorder_obs_in_modality()

  def reorder_obs_in_modality(
      self,
      obs_ordered_index: Optional[pd.Index] = None) -> None:
    """Reorder the AnnData in each modality so the order of obs
    DataFrame in each modality becomes the same.

    Args:
      obs_ordered_index (Optional[pd.Index], optional): the ordered 
      index of obs DataFrame. If not provided, the order of the obs 
      DataFrame of the first (alphabetical order) modality will be used.
      Here, only obs index that exists in all the modalities will be
      used. Defaults to None.

    Returns:
      Dict[str, anndata.AnnData]: ordered dictionary of AnnData.
    """
    for modality in self.data.values():
      adata = modality.adata
      if obs_ordered_index is None:
        obs_ordered_index = adata.obs.index
      else: 
        obs_ordered_index = obs_ordered_index.intersection(adata.obs.index)

    for modality in self.data.values():
      modality.reorder_adata(obs_ordered_index)
        
    return

  @classmethod
  def from_config(cls, config: Dict[str, Any]) -> None:
    all_modality_config = config.get(Constants.CONFIG_NAME_MODALITY)
    modality_list = []
    for modality_config in all_modality_config:
      modality_name = modality_config['name']
      modality_list.append(Modality.from_config(modality_name, config))
    modality_list.sort()

    data = OrderedDict()
    for modality in modality_list:
      data[modality.name] = modality
    
    return cls(data)
