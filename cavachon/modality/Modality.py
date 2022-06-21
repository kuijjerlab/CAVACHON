from __future__ import annotations
from anndata import AnnData
from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.preprocess.Preprocess import Preprocess
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import List

import pandas as pd
import warnings

class Modality:
  def __init__(self, name, modality_type, dist_cls, order, adata, preprocess_steps):
    self.modality_type: str = modality_type
    self.dist_cls: DistributionWrapper = dist_cls
    self.order: int = order
    self.name: str = name
    self.adata: AnnData = adata
    self.preprocess_steps: List[Preprocess] = preprocess_steps
  
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
    return f"Modality {self.order:>02}: {self.name} ({self.modality_type})"
  
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
  def from_config_parser(cls, modality_name, cp: ConfigParser) -> None:
    config = cp.config_modality.get(modality_name)
    modality_name = config.get('name')
    modality_type = config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)
    order = config.get(Constants.CONFIG_FIELD_MODALITY_ORDER)
    dist_cls_name = config.get(Constants.CONFIG_FIELD_MODALITY_DIST)
    dist_cls = ReflectionHandler.get_class_by_name(dist_cls_name)
    adata = FileReader.read_multiomics_data(config, modality_name)
    preprocess_config = config.get('modality')
    preprocess_steps = None

    return cls(
        name=modality_name,
        modality_type=modality_type,
        dist_cls=dist_cls,
        order=order,
        adata=adata,
        preprocess_steps=preprocess_steps)