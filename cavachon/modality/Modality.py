from __future__ import annotations
from anndata import AnnData
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.utils.AnnDataUtils import AnnDataUtils
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import List

import pandas as pd
import warnings

class Modality:
  def __init__(
      self,
      name: str,
      modality_type: str,
      dist_wrapper: DistributionWrapper,
      order,
      adata,
      n_layers,
      n_clusters,
      n_latent_dims):
    self.name: str = name
    self.modality_type: str = modality_type
    self.dist_wrapper: DistributionWrapper = dist_wrapper
    self.order: int = order
    self.adata: AnnData = adata
    self.n_layers: int = n_layers
    self.n_clusters: int = n_clusters
    self.n_latent_dims: int = n_latent_dims
    self.n_obs: int = adata.n_obs
    self.n_vars: int = adata.n_vars
  
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
    """TODO DEPRECATED, can be removed"""
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
    dist_wrapper_name = config.get(Constants.CONFIG_FIELD_MODALITY_DIST)
    dist_wrapper = ReflectionHandler.get_class_by_name(dist_wrapper_name)
    adata = FileReader.read_multiomics_data(cp, modality_name)
    n_layers = config.get(Constants.CONFIG_FIELD_MODALITY_N_LAYERS)
    n_clusters = config.get(Constants.CONFIG_FIELD_MODALITY_N_CLUSTERS)
    n_latent_dims = config.get(Constants.CONFIG_FIELD_MODALITY_N_LATENT_DIMS)

    return cls(
        name=modality_name,
        modality_type=modality_type,
        dist_wrapper=dist_wrapper,
        order=order,
        adata=adata,
        n_layers=n_layers,
        n_clusters=n_clusters,
        n_latent_dims=n_latent_dims)