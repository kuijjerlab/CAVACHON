from __future__ import annotations
from anndata import AnnData
from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Any, Dict

import pandas as pd
import warnings

class Modality:
  def __init__(self, name, data_type, dist_cls, order, adata):
    self.data_type: str = data_type
    self.dist_cls: DistributionWrapper = dist_cls
    self.order: int = order
    self.name: str = name
    self.adata: AnnData = adata
  
  def __lt__(self, other: Modality) -> bool:
    """Overwriten __lt__ function, so Modality can be sorted.

    Args:
        other (Modality): other object to be compared with.

    Returns:
        bool: True if the order of self is smaller than the one from 
        other.
    """
    return self.order < other.order

  def __str__(self) -> str:
    return f"Modality {self.order:>02}: {self.name} ({self.data_type})"
  
  def set_adata(self, adata: AnnData) -> None:
    if not isinstance(adata, AnnData):
      message = f"adata is not an AnnData object, do nothing."
      warnings.warn(message, RuntimeWarning)
      return
    
    self.adata = adata
    return
  
  def reorder_adata(self, obs_ordered_index: pd.Index) -> None:
    """Reorder the AnnData of the modality so the order of obs DataFrame 
    in the AnnData is the same as the provided one.

    Args:
      obs_ordered_index (pd.Index): the desired order of index for the
      obs DataFrame.
    """
    if not isinstance(self.adata, AnnData):
      message = f"{self.__repr__}.adata is not an AnnData object, do nothing."
      warnings.warn(message, RuntimeWarning)
      return
    
    obs_df = self.adata.obs
    var_df = self.adata.var
    matrix = self.adata.X
    n_obs = obs_df.shape[0]
    indices = pd.DataFrame(
      {'IntegerIndex': range(0, n_obs)},
      index=obs_df.index
    )

    reordered_indices = indices.loc[obs_ordered_index, 'IntegerIndex'].values
    reordered_adata = AnnData(X=matrix[reordered_indices])
    reordered_adata.obs = obs_df.iloc[reordered_indices]
    reordered_adata.var = var_df

    self.adata = reordered_adata
    return

  @classmethod
  def from_config(cls, modality, config: Dict[str, Any]) -> None:
    
    all_modality_config = config.get(Constants.CONFIG_NAME_MODALITY)
    for modality_config in all_modality_config:
      if modality_config['name'] == modality:
        modality_name = modality_config['name']
        data_type = modality_config['data_type']
        order = modality_config['order']
        dist_cls_name = modality_config['dist']
        if not dist_cls_name.endswith('Wrapper'):
          dist_cls_name += 'Wrapper'
        dist_cls = ReflectionHandler.get_class_by_name(dist_cls_name)
        adata = FileReader.read_multiomics_data(config, modality_name)

        return cls(name=modality_name, data_type=data_type, dist_cls=dist_cls, order=order, adata=adata)