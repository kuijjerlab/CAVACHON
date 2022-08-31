from __future__ import annotations
from anndata import AnnData
from typing import Any, Dict, List, Optional, Union
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack

import os
import pandas as pd
import yaml

class FileReader:

  @staticmethod
  def read_multiomics_data(config: Config, modality_name: str) -> AnnData:
    """Read (single-cell) multi-omics data for the given modality with 
    the loaded ConfigParser.

    Args:
        cp (ConfigParser): the loaded ConfigParser.

        modality_name (str): the modality to read. 

    Returns:
        AnnData: multi-omics data in AnnData format.
    """
    datadir = config.datadir
    config_modality = config.modality.get(modality_name, {})
    config_sample_list = config_modality.get('samples', [])

    obs_df_list = []
    var_df_list = []
    matrix_list = []
    for sample_name in config_sample_list:
      config_sample = config.sample.get(sample_name)
      config_sample_modality = list(filter(
          lambda x: x.get('name') == modality_name,
          config_sample.get('modalities'))).pop()
      sample_description = config_sample.get('description')
      obs_df = FileReader.read_table(
          filename=os.path.join(datadir, config_sample_modality['barcodes']),
          name=modality_name,
          colnames=config_sample_modality['barcodes_colnames'])
      obs_df['Sample'] = sample_name
      obs_df['Description'] = sample_description

      var_df = FileReader.read_table(
          filename=os.path.join(datadir, config_sample_modality['features']),
          name=modality_name,
          colnames=config_sample_modality['features_colnames'])
      
      matrix = FileReader.read_mtx(os.path.join(datadir, config_sample_modality['matrix']))
      obs_df_list.append(obs_df)
      var_df_list.append(var_df)
      matrix_list.append(matrix)
    
    modality_obs_df = pd.concat(obs_df_list, axis=0)
    modality_var_df = pd.concat(var_df_list, axis=0)
    modality_matrix = vstack(matrix_list)
      
    adata = AnnData(X=modality_matrix)
    adata.obs = modality_obs_df
    adata.var = modality_var_df
  
    return adata

  @staticmethod
  def read_table(
      filename: str,
      name: str,
      colnames: List[str],
      delimiter: str = '\t',
      index_col: Optional[Union[str, int]] = 0,
      **kwargs) -> pd.DataFrame:
    """Read the table as pd.DataFrame.

    Args:
      filename (str): the path of the file to be read.
      
      name (str): the header for the name (name:index_name) for the 
      index of the DataFrame.
      
      colnames (List[str]): the column names of the DataFrame.
      
      delimiter (str, optional): delimiter for the file. Defaults to 
      '\t'.
      
      index_col (Optional[Union[str, int]], optional): the name (or 
      index) for the column used as index (for 
      DataFrame.index). Defaults to 0.

      **kwargs: additional arguments pass to pd.read_csv.

    Returns:
      pd.DataFrame: table in pd.DataFrame.
    """
    filename = os.path.realpath(filename)
    df = pd.read_csv(filename, sep=delimiter, header=None, **kwargs)
    df.columns = colnames
    if not isinstance(index_col, int):
      index_col = colnames.index(index_col)
    index_name = colnames[int(index_col)]
    df.index = pd.Index(df[index_name], name=f"{name}:{index_name}")

    return df

  @staticmethod
  def read_mtx(filename: str) -> csr_matrix:
    """Read a given mtx file, and return the CSR matrix.

    Args:
      filename (str): the filename of the mtx file

    Returns:
      csr_matrix: the content of the mtx file in CSR matrix format.
    """
    filename = os.path.realpath(filename)
    matrix = mmread(filename).transpose().tocsr()
    return matrix

  @staticmethod
  def read_yaml(filename: str) -> Dict[str, Any]:
    """Read a given yaml file, and return the content.

    Args:
      filename (str): the filename of the yaml file.

    Returns:
      Dict[str, Any]: the content of the yaml file.
    """
    content: Dict[str, Any] = dict()
    filename = os.path.realpath(filename)
    with open(filename, 'r') as f:
      content = yaml.load(f, Loader=yaml.FullLoader)
    
    return content
