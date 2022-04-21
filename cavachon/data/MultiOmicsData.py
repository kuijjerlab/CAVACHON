import anndata
import muon as mu
import os
import pandas as pd
import yaml

from cavachon.utils import AnnDataUtils
from collections import defaultdict
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack
from typing import Dict, List, Set

class MultiOmicsData:
  """MultiOmicsData
  Used to read the (single-cell) multi-omics data (obs, var and matrix files). This data
  structure is used as a intermediate layer which read the data from the storage and can
  create a dictionary of AnnData or MuData which can be used to create the Tensorflow 
  dataset. Note that this data structure should only be used to create DataLoader.

  Attributes:
    modalities (Set[str]): the label for the modalities recorded in MultiOmicsData.

    n_samples (int): number of samples in MultiOmicsData.

    sample_index_dict (Dict[str, int]): dictionary of sample index, where the keys are 
    the sample names, and values are the indices to access the data of the sample in 
    `modality_obs_df_dict`, `modality_var_df_dict` and `modality_matrix_df_dict`.

    modality_obs_df_dict (Dict[str, List[pd.DataFrame]]): dictionary of list of obs 
    DataFrame, where keys are the modality, values are list of obs DataFrame of the 
    samples (the mapping between sample and indices is stored in `sample_index_dict`)

    modality_var_df_dict (Dict[str, List[pd.DataFrame]]): dictionary of list of var 
    DataFrame, where keys are the modality, values are list of var DataFrame of the 
    samples (the mapping between sample and indices is stored in `sample_index_dict`)

    modality_obs_df_dict (Dict[str, List[csr_matrix]]): dictionary of list of data matrix 
    , where keys are the modality, values are list of data matrix of the samples (the 
    mapping between sample and indices is stored in `sample_index_dict`)
 
  """
  
  def __init__(self):
    self.modalities: Set[str] = set()
    self.n_samples: int = 0
    self.sample_index_dict: Dict[str, int] = dict()
    self.modality_obs_df_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    self.modality_var_df_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    self.modality_matrix_dict: Dict[str, List[csr_matrix]] = defaultdict(list)

  def add_from_meta(self, meta_path: str) -> None:
    """Add multiple samples from the meta data specification (meta.yaml)

    Args:
        meta_path (str): path to the meta data specification (meta.yaml)
    """
    meta_path = os.path.realpath(meta_path)
    datadir = os.path.dirname(meta_path)

    with open(meta_path) as f:
      specification = yaml.load(f, Loader=yaml.SafeLoader)
    
    for sp_sample in specification['samples']:  
      self.add_sample(datadir, sp_sample)

  def add_sample(self, datadir: str, sp_sample: str) -> None:
    """Add one sample from the meta data specification of one sample.

    Args:
        datadir (str): path to directory where the files (var, obs and matrix) are 
        stored.

        sp_sample (str): the specification of one sample.
    """
    sample = sp_sample['name']
    description = sp_sample['description']
    
    if sample in self.sample_index_dict:
      sample = f"{sample}-{self.n_samples}"
    self.sample_index_dict.setdefault(sample, self.n_samples)

    for sp_modality in sp_sample['modalities']:
      modality = sp_modality['name'].lower()
      self.modalities.add(modality)

      obs_df = MultiOmicsData.read_annot_file(
          datadir, sp_modality['barcodes'], modality, sp_modality['barcodes_cols'])
      obs_df['Sample'] = sample
      obs_df['Description'] = description

      var_df = MultiOmicsData.read_annot_file(
          datadir, sp_modality['features'], modality, sp_modality['features_cols'])

      matrix_path = os.path.join(datadir, sp_modality['matrix'])
      matrix = mmread(matrix_path).transpose().tocsr()

      self.modality_obs_df_dict[modality].append(obs_df)
      self.modality_var_df_dict[modality].append(var_df)
      self.modality_matrix_dict[modality].append(matrix)
    
    self.n_samples += 1
    
    return

  def remove_sample(self, sample: str) -> None:
    """Remove a sample from the MultiOmicsData.

    Args:
        sample (str): the sample name that needs to be removed.
    """
    sample_index = self.sample_index_dict.pop(sample, -1)
    if sample_index < 0:
      return
    
    for modality in self.modalities:
      del self.modality_obs_df[modality][sample_index]
      del self.modality_var_df[modality][sample_index]
      del self.modality_matrix_dict[modality][sample_index]
    
    self.n_samples -= 1

    for sample in self.sample_index_dict:
      if self.sample_index_dict[sample] > sample_index:
        self.sample_index_dict[sample] -= 1

    return 

  def export_adata_dict(self) -> Dict[str, anndata.AnnData]:
    """Export the MultiOmicsData as dictionary of AnnData. The obs DataFrame of the all
    the AnnData are ordered in the same way.

    Returns:
      Dict[str, anndata.AnnData]: exported dictionary of AnnData, where the keys are the
      modality, and the values are the corresponding AnnData.
    """
    modality_adata_dict = {}
    for modality in self.modalities:
      modality_obs_df = pd.concat(self.modality_obs_df_dict[modality], axis=0)
      modality_var_df = pd.concat(self.modality_var_df_dict[modality], axis=0)
      modality_matrix = vstack(self.modality_matrix_dict[modality])
      
      adata = anndata.AnnData(X=modality_matrix)
      adata.obs = modality_obs_df
      adata.var = modality_var_df
      modality_adata_dict.setdefault(modality, adata)
    
    modality_adata_dict = AnnDataUtils.reorder_adata_dict(modality_adata_dict)
    return modality_adata_dict

  def export_mudata(self) -> mu.MuData:
    """Export the MultiOmicsData as MuData. The obs DataFrame of the all modalities are 
    ordered in the same way.

    Returns:
        mu.MuData: exported MuData.
    """
    adata_dict = self.export_adata_dict()
    mdata = mu.MuData(adata_dict)
    mdata.update()

    return mdata

  @staticmethod
  def read_annot_file(
      datadir: str,
      filename: str,
      modality: str,
      colnames: List[str],
      index_col: int = 0) -> pd.DataFrame:
    """Read the annotation files for (single-cell) multi-omics data (DataFrame of obs 
    and var).

    Args:
        datadir (str): the path to the directory where the files are stored.

        filename (str): the filename to be read.
        
        modality (str): the modality of the multi-omics data.
        
        colnames (List[str]): the column names of the DataFrame.
        
        index_col (int, optional): the index column (for DataFrame.index). Defaults to 0.

    Returns:
        pd.DataFrame: annotation DataFrame (obs or var)
    """
    path = os.path.join(datadir, filename)
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = colnames
    index_colname = colnames[index_col]
    df.index = pd.Index(df[index_colname], name=f"{modality}:{index_colname}")

    return df