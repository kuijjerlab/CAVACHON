import pandas as pd

from collections import OrderedDict
from cavachon.modality.Modality import Modality
from cavachon.parser.ConfigParser import ConfigParser
from muon import MuData
from typing import Optional

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
      modality.reorder_or_filter_adata_obs(obs_ordered_index)
        
    return

  def preprocess(self) -> None:
    for modality_name, modality in self.data.items():
      modality.preprocess()
    
    self.reorder_obs_in_modality()
    return

  @classmethod
  def from_config_parser(cls, cp: ConfigParser) -> None:
    data = OrderedDict()
    for modality_name, config in cp.config_modality.items():
      modality = Modality.from_config_parser(modality_name, cp)
      data.setdefault(modality.name, modality)

    return cls(data)

  def export_mudata(self) -> MuData:
    """Export the MultiOmicsData as MuData. The obs DataFrame of the all
    modalities are ordered in the same way.

    Returns:
      mu.MuData: exported MuData.
    """
    self.reorder_obs_in_modality()
    adata_dict = dict()
    for modality_name, modality in self.data.items():
      adata_dict[modality_name] = modality.adata
    mdata = MuData(adata_dict)
    mdata.update()

    return mdata
