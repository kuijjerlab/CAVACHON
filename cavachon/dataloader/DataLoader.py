from __future__ import annotations

import muon as mu
import os
import tensorflow as tf

from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from cavachon.utils.TensorUtils import TensorUtils
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Iterator, Mapping, Optional

class DataLoader:
  """DataLoader
  [TODO: DEPRECATED DOCUMENTATION]
  Data loader to create Tensorflow dataset from MuData.

  Attributes:
    batch_effect_encoder (Dict[str, LabelEncoder]): the encoders used to
    create one-hot encoded batch effect tensor. The keys of the 
    dictionary are formatted as "{modality}:{obs_column}". The 
    LabelEncoder stored the mapping between categorical batch effect
    variables and the numerical representation.

    dataset (tf.data.Dataset): Tensorflow Dataset created from the 
    MuData. Can be used to train/test/validate the CAVACHON model. The 
    field of the dataset includes "{modality}:matrix" (tf.SparseTensor),
    "{modality}:libsize" (tf.Tensor) and "{modality:batch_effect}" 
    (tf.Tensor)

    mdata (mu.MuData): (Single-cell) multi-omics data stored in MuData 
    format.
 
  """
  def __init__(
      self,
      mdata: mu.MuData,
      batch_size: int = 256) -> None:
    self.batch_effect_encoder: Dict[str, Dict[str, LabelEncoder]] = defaultdict(dict)
    self.batch_size: int = batch_size
    self.mdata: mu.MuData = mdata
    self.dataset: Optional[tf.data.Dataset] = None
    
    self.create_dataset()
    self.modify_dataset()

    return

  def create_dataset(self) -> tf.data.Dataset:
    """Create a Tensorflow Dataset based on the MuData provided in the 
    __init__ function.

    Returns:
      tf.data.Dataset: created Dataset. The field of the dataset 
      includes "{modality_name}/matrix" (tf.SparseTensor) and 
      "{modality_name}/batch_effect" (tf.Tensor)
    """
    tensor_mapping = dict()
    modality_names = self.mdata.mod.keys()
    for modality_name in modality_names:
      adata = self.mdata[modality_name]
      data_tensor = TensorUtils.spmatrix_to_sparse_tensor(adata.X)

      # if batch_effect colname is not specified for the current modality, use zero 
      # matrix as batch effect
      if issubclass(type(adata.uns), Mapping):
        adata_config = adata.uns.get('cavachon/config', {})
        batch_effect_colnames = adata_config.get('batch_effect_colnames', None)
      else:
        batch_effect_colnames = None

      if (batch_effect_colnames is None or len(batch_effect_colnames) == 0):
        batch_effect_tensor = tf.zeros((adata.n_obs, 1))
      else:
        batch_effect_tensor, encoder_mapping = TensorUtils.create_tensor_from_df(
            adata.obs, batch_effect_colnames)
        for colnames, encoder in encoder_mapping.items():
          self.batch_effect_encoder[modality_name][colnames] = encoder

      tensor_mapping.setdefault(
          (modality_name, Constants.TENSOR_NAME_X),
          data_tensor)
      tensor_mapping.setdefault(
          (modality_name, Constants.TENSOR_NAME_BATCH),
          batch_effect_tensor)
    
    self.dataset = tf.data.Dataset.from_tensor_slices(tensor_mapping)

  def modify_dataset(self) -> None:
    modality_names = self.mdata.mod.keys()
    for modality_name in modality_names:
      uns = self.mdata[modality_name].uns
      if issubclass(type(uns), Mapping):
        config = uns.get('cavachon/config', {})
        distribution_name = config.get('distribution', '')
        if not distribution_name:
          break
        modifier_class = ReflectionHandler.get_class_by_name(
            distribution_name,
            'dataloader/modifiers')
        modifier = modifier_class(modality_name=modality_name)
        self.dataset = self.dataset.map(modifier)

    return self.dataset

  def __iter__(self) -> Iterator[tf.data.Dataset]:
    return iter(self.dataset.batch(self.batch_size))
  
  @classmethod
  def from_h5mu(cls, h5mu_path: str) -> DataLoader:
    """Create DataLoader from h5mu file (of MultiModality or MuData). 
    Note that if provided with MuData: (1) the different modalities in 
    the MuData needs to be sorted in a way that the order of obs
    DataFrame needs to be the same. (2) the batch effect columns for 
    each modality need to be stored in
    adata.uns['cavachon/config']['batch_effect_columns'] if the user 
    wish to consider batch effect while using the model. 

    Args:
      h5mu_path (str): path to the h5mu file.

    Returns:
      DataLoader: DataLoader created from h5mu file.
    """
    path = os.path.realpath(h5mu_path)
    mdata = mu.read(path)
    mdata.update()
    return cls(mdata)
  
  def load_dataset(self, datadir: str) -> None:
    """Load Tensorflow Dataset snapshot.

    Args:
      datadir (str): the data directory of created Tensorflow Dataset 
      snapshot.
    """
    datadir = os.path.realpath(datadir)
    self.dataset = tf.data.experimental.load(datadir)
    return

  def save_dataset(self, datadir: str) -> None:
    """Save Tensorflow Dataset to local storage.

    Args:
      datadir (str): directory where the Tensorflow Dataset snapshot 
      will be save.
    """
    datadir = os.path.realpath(datadir)
    os.makedirs(datadir, exist_ok=True)
    tf.data.experimental.save(self.dataset, datadir)
    return
