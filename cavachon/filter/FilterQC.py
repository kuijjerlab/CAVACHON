from cavachon.filter.AnnDataFilter import AnnDataFilter
from cavachon.utils.AnnDataUtils import AnnDataUtils
from cavachon.utils.GeneralUtils import GeneralUtils

import anndata
import copy
import operator
import pandas as pd
import scanpy
import warnings

class FilterQC(AnnDataFilter):
  """FilterQC
  
  Filter for AnnData. Used as an adaptor between the 
  scanpy.pp.calculate_qc_metrics() and the configs. After the quality
  metrics is computed, the filtering will be performed based on the
  proprotion of controlled genes. Note that the preprocessing step will 
  be performed inplace.

  Attributes
  ----------
  name: str
      name for the filtering step.
  
  kwargs: Mapping[str, Any]
      additional parameters used for scanpy.pp.calculate_qc_metrics().
      In addition, 'filter_threshold' could be used to specify the
      filtering thresholds. The values for 'filter_threshold' is a
      mapping with keys 'field' (column name for qc metrics in 
      adata.obs), 'operator' (string representation for python 
      operators) and 'threshold' (float number specifying the 
      proportion used to filter cells)

  """
  def __init__(self, name, **kwargs):
    """Constructor for FilterQC

    Parameters
    ----------
    name: str
        name for the filtering step.
  
    kwargs: Mapping[str, Any]
        additional parameters used for scanpy.pp.calculate_qc_metrics().
        In addition, 'filter_threshold' could be used to specify the
        filtering thresholds. The values for 'filter_threshold' is a
        mapping with keys 'field' (column name for qc metrics in 
        adata.obs), 'operator' (string representation for python
        operators) and 'threshold' (float number specifying the 
        proportion used to filter cells)

    """
    super().__init__(name, **kwargs)

  def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:   
    """Perform preprocessing to provided AnnData.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData to preprocessed.

    Returns
    -------
    anndata.AnnData
        preprocessed AnnData.
    
    Raises
    ------
    RuntimeError
        if no cells left after the filtering.

    """ 
    n_obs = adata.obs.shape[0]
    selected = pd.Series(
        GeneralUtils.duplicate_obj_to_list(True, n_obs),
        index=adata.obs.index
    )

    # the filter_threshold is not recognized in 
    # scanpy.pp.calculate_qc_metrics, so we create a copy of self.args 
    # and pop filter_threshold field.
    kwargs_copy = copy.deepcopy(self.kwargs)
    filter_criteria_list = kwargs_copy.pop('filter_threshold')
    
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

    #adata = AnnDataUtils.reorder_or_filter_adata_obs(adata, obs_index)
    adata = adata[obs_index]
    adata.uns.setdefault('dummy', None)
    adata.uns.pop('dummy', None)

    return adata