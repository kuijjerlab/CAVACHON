from cavachon.modifier.adata.AnnDataFilter import AnnDataFilter
from cavachon.utils.AnnDataUtils import AnnDataUtils
from cavachon.utils.GeneralUtils import GeneralUtils

import anndata
import copy
import operator
import pandas as pd
import scanpy
import warnings

class FilterQC(AnnDataFilter):

  def __init__(self, name, *args, **kwargs):
    super().__init__(name)

  def execute(self, adata: anndata.AnnData) -> anndata.AnnData:    
    n_obs = adata.obs.shape[0]
    selected = pd.Series(
        GeneralUtils.duplicate_obj_to_list(True, n_obs),
        index=adata.obs.index
    )

    # the filter_threshold is not recognized in 
    # scanpy.pp.calculate_qc_metrics, so we create a copy of self.args 
    # and pop filter_threshold field.
    kwargs_copy = copy.deepcopy(self.kwargs)
    filter_criteria_list = kwargs_copy.pop('filter_criteria')
    
    # the index columns will be 'Modality.name:colname', the colname
    # (usually gene name) will be used to check the control variables 
    # (usually mitochondria)
    index_colname = adata.var.index.name.split(':')[-1]

    for qc_var in kwargs_copy.setdefault('qc_vars', ['ERCC', 'MT']):
      adata.var[qc_var] = adata.var[index_colname].str.match(qc_var)
    
    scanpy.pp.calculate_qc_metrics(adata, **kwargs_copy)

    for filter_criteria in filter_criteria_list:
      field = filter_criteria.get('field', None)
      threshold = filter_criteria.get('threshold', 0)
      op = getattr(operator, filter_criteria.get('operator', 'ge'))
      if field not in adata.obs:
        message = "".join((
            f"{field} is not in the obs_df of AnnData, ignore the ",
            f"process in {self.__class__.__name__} ({self.name})."
        ))
        warnings.warn(message, RuntimeWarning)
        continue
      selected &= op(adata.obs[field], threshold)
    
    obs_index = adata.obs.loc[selected].index
    if len(obs_index) == 0:
      message = ''.join((
        'No objects left in the adata, please use a less strict '
        f'filter in {self.__class__.__name__} ({self.name}).'
      ))
      raise RuntimeError(message)

    adata = AnnDataUtils.reorder_or_filter_adata_obs(adata, obs_index)
    
    return adata

#%%
