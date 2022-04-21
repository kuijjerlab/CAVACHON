import anndata
import pandas as pd

from typing import Dict, Optional

class AnnDataUtils:
  """AnnDataUtils
  Utility functions for AnnData
  """

  @staticmethod
  def reorder_adata_dict(
      adata_dict: Dict[str, anndata.AnnData],
      obs_ordered_index: Optional[pd.Index] = None) -> Dict[str, anndata.AnnData]:
    """Reorder the dictionary of AnnData so the order of obs DataFrame in each modality
    is the same.

    Args:
      adata_dict (Dict[str, anndata.AnnData]): dictionary of AnnData, where keys are 
      the modality, values are the corresponding AnnData.
      
      obs_ordered_index (Optional[pd.Index], optional): the ordered index of obs 
      DataFrame. If not provided, the order of the obs DataFrame of the first 
      (alphabetical order) modality will be used. Here, only obs index that exists in 
      all the modalities will be used. Defaults to None.

    Returns:
      Dict[str, anndata.AnnData]: ordered dictionary of AnnData.
    """
    for modality in adata_dict.keys():
      adata = adata_dict[modality]

      if obs_ordered_index is None:
        obs_ordered_index = adata.obs.index
      else: 
        obs_ordered_index = obs_ordered_index.intersection(adata.obs.index)

    for modality in adata_dict.keys():
      adata = adata_dict[modality]
      adata_dict[modality] = AnnDataUtils.reorder_adata(adata, obs_ordered_index)

    return adata_dict

  @staticmethod
  def reorder_adata(
      adata: anndata.AnnData,
      obs_ordered_index: pd.Index) -> anndata.AnnData:
    """Reorder the AnnData so teh order of obs DataFrame in teh AnnData is the same as 
    the provided one.

    Args:
        adata (anndata.AnnData): AnnData of a modality.
        obs_ordered_index (pd.Index): the desired order of index for the obs DataFrame.

    Returns:
        anndata.AnnData: ordered AnnData.
    """
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