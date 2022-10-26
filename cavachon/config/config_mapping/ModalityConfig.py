from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.config.config_mapping.FilterConfig import FilterConfig
from cavachon.environment.Constants import Constants
from cavachon.utils.GeneralUtils import GeneralUtils
from copy import deepcopy
from typing import Any, List, Mapping

class ModalityConfig(ConfigMapping):
  """ModalityConfig

  Config for modality.

  Attributes
  ----------
  name: str
      name of the modality.
  
  samples: List[str]
      names of the samples.

  type: str
      modality type.

  dist: str
      distribution name of the modality.

  h5ad: str
      filename to the h5ad (if not provided with samples)

  filters: List[FilterConfig]
      filter step configs for the modality.

  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for ModalityConfig. 

    Parameters
    ----------
    name: str
        name of the modality.

    type: str
        modality type.

    dist: str, optional
        the data distribution of the modality. Currently supports 
        `'IndependentBernoulli'` and 
        `'IndependentZeroInflatedNegativeBinomial'` 
        (see `cavachon/distributions` for more details). Defaults to:
        1. `'IndependentBernoulli'` for `type:atac`.
        2. `'IndependentZeroInflatedNegativeBinomial'` for `type:rna`.

    samples: List[str]
        names of the samples.

    h5ad: str, optional
        filename to the h5ad (if not provided with samples)

    filters: List[FilterConfig]
        filter step configs for the modality.
    
    batch_effect_colnames: List[str]
        the column names of the batch effects that needs to be corrected.

    """
    # change default values here
    self.name: str
    self.type: str
    self.dist: str
    self.samples: List[str] = list()
    self.h5ad: str = ''
    self.filters: List[FilterConfig] = list()
    self.batch_effect_colnames: List[str] = list()

    # preprocess
    kwargs = deepcopy(kwargs)
    ## name
    name = kwargs.get('name')
    kwargs['name'] = GeneralUtils.tensorflow_compatible_str(name)
  
    ## modality type
    modality_type = kwargs.get(Constants.CONFIG_FIELD_MODALITY_TYPE).lower()
    kwargs[Constants.CONFIG_FIELD_MODALITY_TYPE] = modality_type
    
    ## filters
    if Constants.CONFIG_FIELD_MODALITY_FILTER in kwargs:
      filter_configs = kwargs.get(Constants.CONFIG_FIELD_MODALITY_FILTER)
      filter_configs = [FilterConfig(**x) for x in filter_configs]
      kwargs[Constants.CONFIG_FIELD_MODALITY_FILTER] = filter_configs

    ## dist
    if Constants.CONFIG_FIELD_MODALITY_DIST not in kwargs:
      self.dist = Constants.DEFAULT_MODALITY_DISTRIBUTION.get(modality_type)

    super().__init__(
        kwargs, 
        [
          'name',
          'type',
          'dist',
          'samples', 
          'h5ad',
          'filters',
          'batch_effect_colnames'
        ])
    