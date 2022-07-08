from __future__ import annotations
from cavachon.filter.FilterStep import FilterStep
from cavachon.utils.AnnDataUtils import AnnDataUtils
from cavachon.utils.GeneralUtils import GeneralUtils

import copy
import operator
import pandas as pd
import scanpy
import warnings

class FilterQC(FilterStep):

  def __init__(self, name, args):
    super().__init__(name, args)

  def execute(self, modality: Modality) -> None:    
    n_obs = modality.adata.obs.shape[0]
    self.args['inplace'] = True

    # the filter_threshold is not recognized in 
    # scanpy.pp.calculate_qc_metrics, so we create a copy of self.args 
    # and pop filter_threshold field.
    kwargs_copy = copy.deepcopy(self.args)
    filter_step_list = kwargs_copy.pop('filter_threshold')
    
    # the index columns will be 'Modality.name:colname', the colname 
    # will be used to check the control variables (usually mitochondria)
    index_colname = modality.adata.var.index.name.split(':')[-1]

    for qc_var in kwargs_copy.setdefault('qc_vars', ['ERCC', 'MT']):
      modality.adata.var[qc_var] = modality.adata.var[index_colname].str.match(qc_var)
    
    scanpy.pp.calculate_qc_metrics(modality.adata, **kwargs_copy)
    
    selected = pd.Series(
        GeneralUtils.duplicate_obj_to_list(True, n_obs),
        index=modality.adata.obs.index
    )

    for filter_step in filter_step_list:
      field = filter_step.get('field', None)
      threshold = filter_step.get('threshold', 0)
      op = getattr(operator, filter_step.get('operator', 'ge'))
      if field not in modality.adata.obs:
        message = f"{field} is not in the obs_df, ignore the filtering."
        warnings.warn(message, RuntimeWarning)
        continue
      selected &= op(modality.adata.obs[field], threshold)
    
    obs_index = modality.adata.obs.loc[selected].index
    if len(obs_index) == 0:
      message = 'No objects left in the adata, please use a less strict filter.'
      raise RuntimeError(message)

    modality.adata = AnnDataUtils.reorder_or_filter_adata_obs(modality.adata, obs_index)
    modality.n_obs = modality.adata.n_obs
    modality.n_vars = modality.adata.n_vars
    
    return