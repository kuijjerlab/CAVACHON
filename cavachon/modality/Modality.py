from __future__ import annotations
from anndata import AnnData
from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.preprocess.PreprocessStep import PreprocessStep
from cavachon.utils.AnnDataUtils import AnnDataUtils
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
    self.preprocess_steps: List[PreprocessStep] = preprocess_steps
  
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
  
  def reorder_or_filter_adata_obs(self, obs_index: pd.Index) -> None:
    """Reorder the AnnData of the modality so the order of obs DataFrame 
    in the AnnData is the same as the provided one.

    Args:
      obs_index (pd.Index): the desired order of index for the obs 
      DataFrame for reordering or the kept index for the obs DataFrame 
      for filtering.
    """
    if not isinstance(self.adata, AnnData):
      message = f"{self.__repr__}.adata is not an AnnData object, do nothing."
      warnings.warn(message, RuntimeWarning)
      return
    
    self.adata = AnnDataUtils.reorder_or_filter_adata_obs(self.adata, obs_index)
    return

  @classmethod
  def from_config_parser(cls, modality_name, cp: ConfigParser) -> None:
    config = cp.config_modality.get(modality_name)
    modality_name = config.get('name')
    modality_type = config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)
    order = config.get(Constants.CONFIG_FIELD_MODALITY_ORDER)
    dist_cls_name = config.get(Constants.CONFIG_FIELD_MODALITY_DIST)
    dist_cls = ReflectionHandler.get_class_by_name(dist_cls_name)
    adata = FileReader.read_multiomics_data(cp, modality_name)
    config_preprocess_list = config.get(Constants.CONFIG_FIELD_MODALITY_PREPROCESS)
    
    # TODO: Sanitize this part
    preprocess_steps = []
    for config_preprocess in config_preprocess_list:
      preprocess_step_cls_name = Constants.PREPROCESS_STEP_MAPPING.get(
          config_preprocess.get('func'))
      preprocess_step_cls = ReflectionHandler.get_class_by_name(preprocess_step_cls_name)
      preprocess_steps.append(
          preprocess_step_cls(
              config_preprocess.get('name', ''),
              config_preprocess.get('args', {})))

    return cls(
        name=modality_name,
        modality_type=modality_type,
        dist_cls=dist_cls,
        order=order,
        adata=adata,
        preprocess_steps=preprocess_steps)