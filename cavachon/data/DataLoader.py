import anndata
import muon as mu
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from __future__ import annotations
from collections import defaultdict
from scipy.io import mmread
from scipy.sparse import vstack
from typing import Dict, List, Optional, Union
from yaml.loader import SafeLoader
from sklearn.preprocessing import LabelEncoder

class DataLoader:
  def __init__(self, mdata: mu.MuData):
    self.batch_effect_encoder: Dict[str, LabelEncoder] = dict()
    self.mdata: mu.MuData = mdata
    self.dataset: tf.data.Dataset = self.create_dataset(self.mdata)

  def create_batch_effect_tensor(self, adata, modality, batch_effect_colnames=None):
    n_obs = adata.n_obs
    batch_effect_tensor_list = []
    for batch_effect_colname in batch_effect_colnames:
      if batch_effect_colname  not in adata.obs.columns:
        continue
      batch_target = adata.obs[batch_effect_colname]
      if len(np.unique(batch_target)) < n_obs * 0.2:
        # if the column is a categorical variable, use one hot encoded tensor
        le = LabelEncoder()
        batch_integer = le.fit_transform(batch_target)
        n_class = len(le.classes_)
        self.batch_effect_encoder.setdefault(f"{modality}:{batch_effect_colname}", le)
        batch_effect_tensor_list.append(
            tf.cast(tf.one_hot(batch_integer, n_class), tf.float32)
        )
      else:
        # if the column is a continous variable, 
        batch_effect_tensor_list.append(tf.convert_to_tensor(batch_target))

    if len(batch_effect_tensor_list == 0):
      return tf.zeros((adata.n_obs, 1))
    else:
      return tf.concat(batch_effect_tensor_list, axis=1)

  def create_dataset(
      self,
      libsize_colnames_dict: Dict[str, str] = None,
      batch_effect_colnames_dict: Dict[str, List[str]] = None):
    field_dict = dict()
    for modality in self.mdata.mod.keys():
      adata = self.mdata[modality]
      data_tensor = DataLoader.create_sparse_tensor(adata.X)
      
      if (libsize_colnames_dict is None or 
          modality not in libsize_colnames_dict or
          libsize_colnames_dict[modality] not in adata.obs.columns):
        # if library colname is not specified for the current modality, use the 
        # row sum of the matrix
        libsize = adata.X.sum(axis=1)
      else:
        library_colname = libsize_colnames_dict[modality]
        libsize = np.reshape(adata.obs[library_colname].values, (-1, 1))
      libsize_tensor = tf.convert_to_tensor(libsize, dtype=tf.float32)

      if (batch_effect_colnames_dict is None or 
          modality not in batch_effect_colnames_dict):
        # if library colname is not specified for the current modality, use zero 
        # matrix as batch effect
        batch_effect_tensor = tf.zeros((adata.n_obs, 1))
      else:
        batch_effect_tensor = self.create_batch_effect_tensor(
            adata, modality, batch_effect_colnames_dict[modality]
        )

      field_dict.setdefault(f"{modality}:matrix", data_tensor)
      field_dict.setdefault(f"{modality}:libsize", libsize_tensor)
      field_dict.setdefault(f"{modality}:batch_effect", batch_effect_tensor)
    
    return tf.data.Dataset.from_tensor_slices(field_dict)

  @staticmethod
  def create_sparse_tensor(matrix):  
    coo_matrix = matrix.tocoo()
    indices = np.mat([coo_matrix.row, coo_matrix.col]).transpose()
    return tf.SparseTensor(indices, coo_matrix.data, coo_matrix.shape)

  @classmethod
  def from_dict(cls, adata_dict: Dict[str, anndata.AnnData]) -> DataLoader:
    barcodes = None
    for modality in adata_dict.keys():
      adata = adata_dict[modality]
      barcodes = adata.obs.index if barcodes is None else barcodes
      adata_dict[modality] = DataLoader.reorder_adata(adata, barcodes)

    mdata = mu.MuData(adata_dict)
    mdata.update()
    return cls(mdata)
  
  @classmethod
  def from_h5mu(cls, h5mu_path: str) -> DataLoader:
    path = os.path.realpath(h5mu_path)
    mdata = mu.read(path)
    mdata.update()
    return cls(mdata)

  @classmethod
  def from_meta(cls, meta_path: str) -> DataLoader:
    meta_path = os.path.realpath(meta_path)
    datadir = os.path.dirname(meta_path)

    with open(meta_path) as f:
      specification = yaml.load(f, Loader=SafeLoader)

    modalities = []
    modality_adata_dict = {}
    modality_obs_df_dict = defaultdict(list)
    modality_var_df_dict = defaultdict(list)
    modality_matrix_dict = defaultdict(list)

    for sp_sample in specification['samples']:
      sample = sp_sample['name']
      description = sp_sample['description']

      for sp_modality in sp_sample['modalities']:
        modality = sp_modality['name'].lower()
        modalities.append(modality)

        obs_df = DataLoader.read_annot_file(
          datadir,
          sp_modality['barcodes'],
          modality,
          sp_modality['barcodes_cols']
        )
        obs_df['Sample'] = sample
        obs_df['Description'] = description

        var_df = DataLoader.read_annot_file(
          datadir,
          sp_modality['features'],
          modality,
          sp_modality['features_cols']
        )

        matrix_path = os.path.join(datadir, sp_modality['matrix'])
        matrix = mmread(matrix_path).transpose().tocsr()
        
        modality_obs_df_dict[modality].append(obs_df)
        modality_var_df_dict[modality].append(var_df)
        modality_matrix_dict[modality].append(matrix)
    
    for modality in modalities:
      modality_obs_df = pd.concat(modality_obs_df_dict[modality], axis=0)
      modality_var_df = pd.concat(modality_var_df_dict[modality], axis=0)
      modality_matrix = vstack(modality_matrix_dict[modality])
      
      adata = anndata.AnnData(X=modality_matrix)
      adata.obs = modality_obs_df
      adata.var = modality_var_df
      modality_adata_dict.setdefault(modality, adata)

    return cls.from_dict(modality_adata_dict)
  
  def load_dataset(self, datadir) -> None:
    datadir = os.path.realpath(datadir)
    self.dataset = tf.data.experimental.load(datadir)
    return

  @staticmethod
  def read_annot_file(datadir, filename, modality, colnames, index_col=0):
    path = os.path.join(datadir, filename)
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = colnames
    index_colname = colnames[index_col]
    df.index = pd.Index(df[index_colname], name=f"{modality}:{index_colname}")

    return df 

  @staticmethod
  def reorder_adata(adata, obs_ordered_index):
    obs_df = adata.obs
    var_df = adata.var
    matrix = adata.X
    n_obs = obs_df.shape[0]
    indices = pd.DataFrame(
      {'IntegerIndex': range(0, n_obs)},
      index=obs_df.index
    )

    reordered_indices = indices.loc[obs_ordered_index, 'IntegerIndex'].values

    reordered_adata = anndata.AnnData(X=matrix[reordered_indices])
    reordered_adata.obs = obs_df.iloc[reordered_indices]
    reordered_adata.var = var_df

    return reordered_adata
  
  def save_dataset(self, datadir) -> None:
    datadir = os.path.realpath(datadir)
    os.makedirs(datadir, exist_ok=True)
    tf.data.experimental.save(self.dataset, datadir)
    return

  def save_mdata(self, path) -> None:
    path = os.path.realpath(path)
    self.mdata.write_h5mu(path)
    return