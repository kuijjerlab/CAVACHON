import copy
import os
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.utils.GeneralUtils import GeneralUtils
from collections import OrderedDict
from typing import Any, Dict

class ConfigParser:
  """ConfigParser
  Data structure for config.yaml.

  Attributes:
    config (Dict[str, Any]): all configuration.

    config_io (Dict[str, str]): IO related configuration.

    config_sample (OrderedDict[str, Any]): sample related configuration,
    where the keys are the sample names, and the values are the 
    configuration for the sample.

    config_modality (OrderedDict[str, Any]): modality related
    configuration, where the keys are the modality names, and the values
    are the configuration for the modality.

    datadir (str): the directory of the input data.

    filename (str): the filename of the config yaml.
  """
  
  def __init__(self, filename: str) -> None:
    self.datadir: str = ''
    self.filename = os.path.realpath(filename)
    self.config: Dict[str, Any] = FileReader.read_yaml(filename)
    self.config_io: Dict[str, str] = dict()
    self.config_sample: OrderedDict[str, Any] = OrderedDict()
    self.config_modality: OrderedDict[str, Any] = OrderedDict()
    
    self.setup()

    return

  def setup(self) -> None:
    """Setup the configuration after reading config.yaml."""
    self.setup_iopath()
    self.setup_config_modality()
    self.setup_config_sample()
    return    

  def setup_iopath(self) -> None:
    """Setup IO related directories"""
    self.config_io = self.config.get(Constants.CONFIG_FIELD_IO)
    self.datadir = os.path.realpath(
        os.path.dirname(f'{self.config_io.get(Constants.CONFIG_FIELD_IO_DATADIR)}/'))
    return

  def setup_config_modality(self) -> None:
    """This function does the following:
    1. Setup config for modalities, transform the list of modality 
    config to a OrderedDict where the keys are the modality names, and 
    the values are the configuration for the modality. The order of the
    insertion depends on the order field in the config (in ascending
    order)
    2. Check all the relevant fields are in the config. Set defualts if 
    the optional fields are not provided.
    """
    config_list = self.config.get(Constants.CONFIG_FIELD_MODALITY, [])
    sorted_config_list = sorted(
        config_list, 
        key=lambda x: x.get(Constants.CONFIG_FIELD_MODALITY_ORDER, 0))
    for i, config in enumerate(sorted_config_list):
      all_required_keys_are_there, keys_not_provided = GeneralUtils.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_MODALITY_REQUIRED,
          config)
      modality_name = config.get('name', f"Modality-{i:>02}")
      if not all_required_keys_are_there:
        message = ''
        for key in keys_not_provided:
          message += f'{key} is required in the modality config for {modality_name}.\n'
        raise KeyError(message)
      config.setdefault('samples', [])
      config.setdefault(
          'dist',
          Constants.DEFAULT_DIST[config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)])
      if not config.get('dist').endswith('Wrapper'):
        config['dist'] += 'Wrapper'
      self.config_modality[modality_name] = config

    return

  def setup_config_sample(self) -> None:
    """This function does the following:
    1. Setup config for sample, transform the list of sample config to a 
    OrderedDict where the keys are the sample names, and the values are 
    the configuration for the sample. The order of the insertion depends 
    on the order the samples appear.
    2. Save the sample modality config to each modality in the 
    config_modality.
    3. Check all the relevant fields are in the config. Set defualts if 
    the optional fields are not provided.

    Raises:
      KeyError: raise if the configuration for the modality of any 
      sample cannot be found in config_modality.
    """
    config_list = self.config.get(Constants.CONFIG_FIELD_SAMPLE, [])
    for i, config in enumerate(config_list):
      sample_name = config.get('name', f"Sample-{i:>02}")
      sample_description = config.get('description', f"Sample-{i:>02}")
      self.config_sample[sample_name] = config
      for sample_modality in config.get(Constants.CONFIG_FIELD_SAMPLE_MODALITY, []):
        modality_name = sample_modality.get('name', None)
        if modality_name in self.config_modality:
          updated_sample_modality = copy.deepcopy(sample_modality)
          updated_sample_modality.setdefault('sample_name', sample_name)
          updated_sample_modality.setdefault('sample_description', sample_description)
          self.config_modality[modality_name].get('samples', []).append(updated_sample_modality)
        else:
          message = f"No configuration for modality {modality_name} of sample {sample_name}"
          raise KeyError(message)
        
        all_required_keys_are_there, keys_not_provided = GeneralUtils.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED,
          sample_modality)
        if not all_required_keys_are_there:
          message = ''
          for key in keys_not_provided:
            message += f'{key} is required in the sample config for {sample_name}.\n'
          raise KeyError(message)
