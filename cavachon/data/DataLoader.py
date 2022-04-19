from __future__ import annotations

import anndata
import muon as mu
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from cavachon.data.MultiOmicsData import MultiOmicsData
from cavachon.utils.AnnDataUtils import AnnDataUtils
from cavachon.utils.TensorUtils import TensorUtils
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

class DataLoader:
  """DataLoader
  Data loader to create Tensorflow dataset from MuData.

  Attributes:
    batch_effect_encoder (Dict[str, LabelEncoder]): the encoders used to create one-hot 
    encoded batch effect tensor. The keys of the dictionary are formatted as 
    "{modality}:{obs_column}". The LabelEncoder stored the mapping between categorical 
    batch effect variables and the numerical representation.

    dataset (tf.data.Dataset): Tensorflow Dataset created from the MuData. Can be used 
    to train/test/validate the CAVACHON model. The field of the dataset includes 
    "{modality}:matrix" (tf.SparseTensor), "{modality}:libsize" (tf.Tensor) and 
    "{modality:batch_effect}" (tf.Tensor)

    mdata (mu.MuData): (Single-cell) multi-omics data stored in mu.MuData format.
 
  """
  def __init__(self, mdata: mu.MuData) -> None:
    self.batch_effect_encoder: Dict[str, LabelEncoder] = dict()
    self.mdata: mu.MuData = mdata
    self.dataset: tf.data.Dataset = self.create_dataset(self.mdata)

    return

  def create_dataset(
      self,
      libsize_colnames_dict: Dict[str, str] = None,
      batch_effect_colnames_dict: Dict[str, List[str]] = None) -> tf.data.Dataset:
    """Create a Tensorflow Dataset based on the MuData provided in the __init__ function.


    Args:
      libsize_colnames_dict (Dict[str, str], optional): dictionary of the column name of 
      libsize in the obs DataFrame, where the keys are the modalities, and values are the
      libsize column corresponds to the modality. Note that the libsize column needs to 
      be a continous variable. Defaults to None.

      batch_effect_colnames_dict (Dict[str, List[str]], optional): dictionary of the 
      column name of batch effect in the obs DataFrame, where the keys are the 
      modalities, and values are the batch effect columns (in list) corresponds to the 
      modality. The batch effect columns can be either categorical or continuous. 
      Defaults to None.

    Returns:
      tf.data.Dataset: created Dataset. The field of the dataset includes 
      "{modality}:matrix" (tf.SparseTensor), "{modality}:libsize" (tf.Tensor) and 
      "{modality:batch_effect}" (tf.Tensor)
    """
    field_dict = dict()
    for modality in self.mdata.mod.keys():
      adata = self.mdata[modality]
      data_tensor = TensorUtils.create_sparse_tensor(adata.X)
      
      if (libsize_colnames_dict is None or 
          modality not in libsize_colnames_dict or
          libsize_colnames_dict[modality] not in adata.obs.columns):
        # if library colname is not specified for the current modality, use the 
        # row sum of the matrix
        libsize_tensor = tf.convert_to_tensor(adata.X.sum(axis=1))
      else:
        library_colname = libsize_colnames_dict[modality]
        libsize_tensor, _ = TensorUtils.create_tensor_from_df(
            adata.obs, [library_colname]
        )

      if (batch_effect_colnames_dict is None or 
          modality not in batch_effect_colnames_dict):
        # if library colname is not specified for the current modality, use zero 
        # matrix as batch effect
        batch_effect_tensor = tf.zeros((adata.n_obs, 1))
      else:
        batch_effect_tensor, encoder_dict = TensorUtils.create_tensor_from_df(
            adata.obs, batch_effect_colnames_dict[modality]
        )
        for mod, encoder in encoder_dict.items():
          self.batch_effect_encoder[f"{modality}:{mod}"] = encoder

      field_dict.setdefault(f"{modality}:matrix", data_tensor)
      field_dict.setdefault(f"{modality}:libsize", libsize_tensor)
      field_dict.setdefault(f"{modality}:batch_effect", batch_effect_tensor)
    
    return tf.data.Dataset.from_tensor_slices(field_dict)

  @classmethod
  def from_dict(cls, adata_dict: Dict[str, anndata.AnnData]) -> DataLoader:
    """Create DataLoader from the dictionary of AnnData.

    Args:
      adata_dict (Dict[str, anndata.AnnData]): dictionary of AnnData, where keys are 
      the modality, values are the corresponding AnnData.

    Returns:
      DataLoader: DataLoader created from the dictionary of AnnData.
    """
    adata_dict = AnnDataUtils.reorder_adata_dict(adata_dict)
    mdata = mu.MuData(adata_dict)
    mdata.update()
    return cls(mdata)
  
  @classmethod
  def from_h5mu(cls, h5mu_path: str) -> DataLoader:
    """Create DataLoader from h5mu file (of MuData). Note that the different modalities
    in the MuData needs to be sorted in a way that the order of obs DataFrame needs to 
    be the same.

    Args:
      h5mu_path (str): path to the h5mu file.

    Returns:
      DataLoader: DataLoader created from h5mu file.
    """
    path = os.path.realpath(h5mu_path)
    mdata = mu.read(path)
    mdata.update()
    return cls(mdata)

  @classmethod
  def from_meta(cls, meta_path: str) -> DataLoader:
    """Create DataLoader from meta data specification (meta.yaml).

    Args:
      meta_path (str): path to meta.yaml.

    Returns:
      DataLoader: DataLoader created from meta.yaml.
    """
    data = MultiOmicsData()
    data.add_from_meta(meta_path)
    modality_adata_dict = data.export_adata_dict()
    return cls.from_dict(modality_adata_dict)
  
  def load_dataset(self, datadir: str) -> None:
    """Load Tensorflow Dataset snapshot.

    Args:
      datadir (str): the data directory of created Tensorflow Dataset snapshot.
    """
    datadir = os.path.realpath(datadir)
    self.dataset = tf.data.experimental.load(datadir)
    return

  def save_dataset(self, datadir: str) -> None:
    """Save Tensorflow Dataset to local storage.

    Args:
      datadir (str): directory where the Tensorflow Dataset snapshot will be save.
    """
    datadir = os.path.realpath(datadir)
    os.makedirs(datadir, exist_ok=True)
    tf.data.experimental.save(self.dataset, datadir)
    return

  def save_mdata(self, path: str) -> None:
    """Save MuData to local storage.

    Args:
      path (str): path where the h5mu file of MuData will be save.
    """
    path = os.path.realpath(path)
    self.mdata.write_h5mu(path)
    return
