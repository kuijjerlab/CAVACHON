from cavachon.config.config_mapping.ComponentConfig import ComponentConfig
from cavachon.config.config_mapping.DatasetConfig import DatasetConfig
from cavachon.config.config_mapping.FilterConfig import FilterConfig
from cavachon.config.config_mapping.IOConfig import IOConfig
from cavachon.config.config_mapping.ModalityConfig import ModalityConfig
from cavachon.config.config_mapping.ModelConfig import ModelConfig
from cavachon.config.config_mapping.OptimizerConfig import OptimizerConfig
from cavachon.config.config_mapping.SampleConfig import SampleConfig
from cavachon.config.config_mapping.TrainingConfig import TrainingConfig
from cavachon.environment.Constants import Constants
from cavachon.utils.GeneralUtils import GeneralUtils
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

import os
import yaml

class Config:
  """Config

  Data structure for the configuration for CAVACHON.

  Attributes
  ----------
  filename: str
      filename of the config in YAML format.

  io: IOConfig
      IO related config.
    
  sample: OrderedDict[str, SampleConfig]
      sample related config, where the key is the sample name, the 
      value is the corresponding SampleConfig.
  
  modality: Dict[str, ModalityConfig]
      modality related config, where the key is the modality name, the 
      value is the corresponding ModalityConfig.

  modality_names: List[str]
      all used modality names.

  filter: Dict[str, List[FilterConfig]]
      modality filter steps related config, where the key is the name 
      of the modality to filter, the value is a list of config for 
      filtering steps.

  model: ModelConfig
      model related config.

  training: TrainingConfig
      training related config.
  
  dataset: DatasetConfig
      dataset related config.

  components: List[ComponentConfig]
      the topological sorted (based on dependency graph) list of 
      components related config.

  yaml: Dict[str, Any]
      the original yaml config in dictionary format.

  """

  def __init__(self, filename: str) -> None:
    """Constructor for Config instance.

    Parameters
    ----------
    filenames: str
        filename of the config in YAML format.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.

    See Also
    --------
    setup_io: setup the io config and datadir.
    setup_modality: setup modality related config.
    setup_sample: setup sample related config.
    setup_training: setup training related config.
    setup_dataset: setup dataset related config.
    setup_model: setup model related config.

    """
    self.filename = os.path.realpath(filename)
    with open(filename, 'r') as f:
      self.yaml: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

    # initializations
    self.io: IOConfig = None
    self.sample: OrderedDict[str, SampleConfig] = OrderedDict()
    self.modality: Dict[str, ModalityConfig] = dict()
    self.modality_names: List[str] = list()
    self.model: ModelConfig = None
    self.filter: Dict[str, List[FilterConfig]] = dict()
    self.training: TrainingConfig = None
    self.components: List[ComponentConfig] = list()
    self.dataset: DatasetConfig = None
    
    # set defaults values, preprocessing the configs
    self.setup_io()
    self.setup_modality()
    self.setup_sample()
    self.setup_training()
    self.setup_dataset()
    self.setup_model()

    return

  def are_all_fields_in_mapping(
      self,
      key_list: List[Any],
      mapping: Mapping,
      field: str,
      subfield: str = '') -> bool:
    """Check if all the required keys are in the provided mapping.

    Parameters
    ----------
    key_list: List[Any]:
        the required list of keys.
    
    mapping: Mapping
        the mapping to be evaluated.
    
    field: str
        the field of config (only used for error message)

    subfield: str, optional
        the subfield of config (only used for error message). Defaults 
        to ''.

    Returns
    -------
    bool
        whether all the required keys are in the provided mapping.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.
    
    """
    keys_not_exist = []
    for key in key_list:
      if key not in mapping:
        keys_not_exist.append(key)
    
    all_required_keys_are_there = len(keys_not_exist) == 0
    if not all_required_keys_are_there:
      message = ''
      for key in keys_not_exist:
        if subfield != '':
          subfield = f' in the {subfield}'
        message += ''.join((
            f'{key} is required{subfield} for ',
            f'{field} in the config file {self.filename}.\n'))
      raise KeyError(message)

    return all_required_keys_are_there
  
  def setup_io(self) -> None:
    self.io = IOConfig(**self.yaml.get(Constants.CONFIG_FIELD_IO))
 
  def setup_modality(self) -> None:
    """Setup modality related config and modality names. This function
    does the following:
    1. Setup config for modality, transform the list of modality config 
       to a dictionary where the keys are the modality names, and the 
       values are the configuration for the modalities.
    2. Check all the relevant fields are in the config. Set defaults if 
       the optional fields are not provided.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.
    
    """
    # Clear self.modality and self.filter
    self.modality = dict()
    self.filter = dict()

    # Get the list of modality config
    modality_config_list = self.yaml.get(Constants.CONFIG_FIELD_MODALITY, [])
    if len(modality_config_list) == 0:
      raise KeyError(f'No modality found in the config file {self.filename}.')

    for i, modality_config in enumerate(modality_config_list):
      # Check if all required keys are in the modalilty config.
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_MODALITY_REQUIRED,
          modality_config,
          Constants.CONFIG_FIELD_MODALITY)
        
      # Create the ModalityConfig instance
      modality_config = ModalityConfig(**modality_config)
      
      # Save the processed result to self.modality and self.filter
      modality_name = modality_config.name
      filters_config = modality_config.filters
      self.modality.setdefault(modality_name, modality_config)
      self.filter.setdefault(modality_name, filters_config)
    
    # Save the modality names
    self.modality_names = list(self.modality.keys())
      
    return

  def setup_sample(self) -> None:
    """Setup sample related config. This function does the following:
    1. Setup config for sample, transform the list of sample config to 
       a OrderedDict where the keys are the sample names, and the 
       values are the configuration for the sample. The order of the 
       insertion depends on the order the samples appear.
    2. Save the sample modality config to each modality in the 
       config_modality.
    3. Check all the relevant fields are in the config. Set defaults if 
       the optional fields are not provided.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.
    
    """
    # Clear the OrderedDict of config sample
    self.sample = OrderedDict()
    # Get the list of sample config
    sample_config_list = self.yaml.get(Constants.CONFIG_FIELD_SAMPLE, [])
    for i, sample_config in enumerate(sample_config_list):
      # Check if all required keys are in the sample config.
      sample_name = sample_config.get('name')
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_SAMPLE_REQUIRED,
          sample_config,
          Constants.CONFIG_FIELD_SAMPLE)

      # Check each modality config in the sample
      for i, sample_modality_config in enumerate(sample_config.get(Constants.CONFIG_FIELD_MODALITY)):
        # Check if all required keys are in the modalilty of the sample.
        self.are_all_fields_in_mapping(
            Constants.CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED,
            sample_modality_config,
            sample_name,
            Constants.CONFIG_FIELD_MODALITY)

        # Check if the modality name is in the self.modality
        modality_name = sample_modality_config.get('name')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
        if modality_name in self.modality:
          # Put the sample names into self.modality[`modality_name`][`sample`]
          self.modality.get(modality_name).get(Constants.CONFIG_FIELD_SAMPLE).append(sample_name)
        else:
          message = "".join((
            f"No configuration for modality {modality_name} of sample {sample_name} ",
            f"in the config file {self.filename}."))
          raise KeyError(message)

    for i, sample_config in enumerate(sample_config_list):
      self.sample.setdefault(sample_name, SampleConfig(**sample_config))   
    
    return

  def setup_training(self) -> None:
    """Setup training related config."""
    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    training_config = model_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING, {})
    self.training = TrainingConfig(**training_config)

  def setup_dataset(self) -> None:
    """Setup dataset related config."""
    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    dataset_config = model_config.get(Constants.CONFIG_FIELD_MODEL_DATASET, {})
    self.dataset = DatasetConfig(**dataset_config)

  def setup_model(self) -> None:
    """Setup model and component related config.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.

    AttributeError
        if the dependencies between components is not a directed 
        acyclic graph.

    """
    self.model = None
    self.components = list()
    
    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    self.are_all_fields_in_mapping(
        Constants.CONFIG_FIELD_MODEL_REQUIRED,
        model_config,
        'model config')

    # Check required field and default values if not specified in the component config
    component_config_list = model_config.get(Constants.CONFIG_FIELD_MODEL_COMPONENT)
    component_config_mapping = dict()
    for i, component_config in enumerate(component_config_list):
      # Check required field
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_COMPONENT_REQUIRED,
          component_config,
          'component')
      component_name = component_config.get('name')

      # Setup modality names and decoder for the component
      for modality_config in component_config.get(Constants.CONFIG_FIELD_MODALITY):
        self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_COMPONENT_MODALITIES_REQUIRED,
          modality_config,
          component_name,
          'modalities config')
        modality_name = modality_config.get('name')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
        modality_dist = self.modality.get(modality_name).get(Constants.CONFIG_FIELD_MODALITY_DIST)
        modality_config[Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES] = modality_dist
      
      component_config = ComponentConfig(**component_config)
      component_config_mapping.setdefault(component_name, component_config)
    
    # Sort the components based on the BFS order
    self.components = GeneralUtils.order_components(component_config_mapping)
    
    model_fields = [
        'name',
        Constants.CONFIG_FIELD_MODEL_LOAD_WEIGHTS,
        Constants.CONFIG_FIELD_MODEL_SAVE_WEIGHTS,
        Constants.CONFIG_FIELD_MODEL_COMPONENT,
        Constants.CONFIG_FIELD_MODEL_TRAINING,
        Constants.CONFIG_FIELD_MODEL_DATASET
    ]

    self.model = ModelConfig(**{field: model_config.get(field) for field in model_fields})