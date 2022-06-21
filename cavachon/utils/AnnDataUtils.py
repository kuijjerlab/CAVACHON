import pandas as pd

from anndata import AnnData
from typing import Dict, Optional
import warnings

class AnnDataUtils:
  """AnnDataUtils
  Utility functions for AnnData
  """

  @staticmethod
  def reorder_or_filter_adata_obs(
      adata: AnnData,
      obs_index: pd.Index) -> AnnData:
    """Reorder the AnnData so teh order of obs DataFrame in teh AnnData
    is the same as the provided one.
    Args:
      adata (anndata.AnnData): AnnData to be reordered (or filtered).
        
      obs_index (pd.Index): the desired order of index for the obs 
      DataFrame for reordering or the kept index for the obs DataFrame 
      for filtering.
    Returns:
        anndata.AnnData: ordered AnnData.
    """
    if not isinstance(adata, AnnData):
      message = "Provided adata is not an AnnData object, do nothing."
      warnings.warn(message, RuntimeWarning)
      return

    obs_df = adata.obs
    var_df = adata.var
    matrix = adata.X
    n_obs = obs_df.shape[0]
    indices = pd.DataFrame(
        {'IntegerIndex': range(0, n_obs)},
        index=obs_df.index
    )

    selected_indices = indices.loc[obs_index, 'IntegerIndex'].values

    selected_adata = AnnData(X=matrix[selected_indices])
    selected_adata.obs = obs_df.iloc[selected_indices]
    selected_adata.var = var_df

    return selected_adata