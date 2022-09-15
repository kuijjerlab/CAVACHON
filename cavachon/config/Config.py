import os
from cavachon.config.ComponentConfig import ComponentConfig
from cavachon.config.DatasetConfig import DatasetConfig
from cavachon.config.FilterConfig import FilterConfig
from cavachon.config.IOConfig import IOConfig
from cavachon.config.ModalityConfig import ModalityConfig
from cavachon.config.ModelConfig import ModelConfig
from cavachon.config.OptimizerConfig import OptimizerConfig
from cavachon.config.SampleConfig import SampleConfig
from cavachon.config.TrainingConfig import TrainingConfig
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.utils.GeneralUtils import GeneralUtils
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

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

  def __init__(
      self,
      filename: str,
      default_data_dir: str = './',
      default_optimizer: str = 'adam',
      default_learning_rate: float = 5e-4,
      default_max_n_epochs: int = 500,
      default_batch_size: int = 256,
      default_n_latent_dims: int = 5,
      default_n_encoder_layers: int = 3,
      default_n_decoder_layers: int = 3,
      default_n_progressive_epochs: int = 100) -> None:
    """Constructor for Config instance.

    Parameters
    ----------
    filenames: str
        filename of the config in YAML format.

    default_data_dir: str, optional
        default value for the data directory. Defaults to './'.
    
    default_optimizer: str, optional
        default optimizer. Defaults to 'adam'.
    
    default_learning_rate: float, optional
        default learning rate for optimizer. Defaults to 5e-4.

    default_max_n_epochs: int, optional
        default maximum number of epochs for training. Defaults to 500.

    default_batch_size: int, optional
        default batch size for training and predict. Defaults to 128.

    default_n_latent_dims: int, optional
        default number of latent dimensions. Defaults to 5.

    default_n_encoder_layers: int, optional
        default number of layers in the encoder backbone. Defaults to 3.

    default_n_decoder_layers: int, optional
        default number of layers in the decoder backbone. Defaults to 3.
    
    default_n_progressive_epochs: int, optional
        default number of progressive epochs. Default to 100.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.

    See Also
    --------
    setup_iopath: setup the io config and datadir.
    setup_config_modality: setup modality related config.
    setup_training_dataset: setup training and dataset related config.
    setup_config_model: setup model related config.

    """
    self.filename = os.path.realpath(filename)
    self.yaml: Dict[str, Any] = FileReader.read_yaml(filename)
    self.io: IOConfig = IOConfig(dict())
    self.sample: OrderedDict[str, SampleConfig] = OrderedDict()
    self.modality: Dict[str, ModalityConfig] = dict()
    self.modality_names: List[str] = list()
    self.model: ModelConfig = ModelConfig(dict())
    self.filter: Dict[str, List[FilterConfig]] = dict()
    self.training: TrainingConfig = TrainingConfig(dict())
    self.components: List[ComponentConfig] = list()
    self.dataset: DatasetConfig = DatasetConfig(dict())
    
    self.setup_iopath(default_datadir=default_data_dir)
    self.setup_config_modality()
    self.setup_config_sample()
    self.setup_training_dataset(
        default_optimizer=default_optimizer,
        default_learning_rate=default_learning_rate,
        default_max_n_epochs=default_max_n_epochs,
        default_batch_size=default_batch_size)
    self.setup_config_model(
        default_n_latent_dims=default_n_latent_dims,
        default_n_encoder_layers=default_n_encoder_layers,
        default_n_decoder_layers=default_n_decoder_layers,
        default_n_progressive_epochs=default_n_progressive_epochs)

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

  def setup_iopath(self, default_datadir: str = './') -> None:
    """Setup the io config and datadir.

    Parameters
    ----------
    default_data_dir: str, optional
        default value for the data directory. Defaults to './'.

    """
    io_config = self.yaml.get(Constants.CONFIG_FIELD_IO)
    datadir = io_config.get(Constants.CONFIG_FIELD_IO_DATADIR, default_datadir)
    io_config[Constants.CONFIG_FIELD_IO_DATADIR] = os.path.realpath(os.path.dirname(f'{datadir}/'))

    self.io = IOConfig(io_config)
    return

  def setup_config_modality(self) -> None:
    """Setup modality related config and modality names.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.
    
    """
    # Clear self.modality and self.modality_filter
    self.modality = dict()
    self.filter = dict()

    # Get the list of modality config
    modality_config_list = self.yaml.get(Constants.CONFIG_FIELD_MODALITY, [])
    if len(modality_config_list) == 0:
      raise KeyError(f'No modality found in the config file {self.filename}.')

    for i, modality_config in enumerate(modality_config_list):
      # Check if all required keys are in the modalilty config.
      modality_name = modality_config.get('name', f"modality/{i:02d}")
      modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
      modality_config['name'] = modality_name
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_MODALITY_REQUIRED,
          modality_config,
          Constants.CONFIG_FIELD_MODALITY)
      modality_type = modality_config.get(Constants.CONFIG_FIELD_MODALITY_TYPE).lower()
      filter_configs = modality_config.get(Constants.CONFIG_FIELD_MODALITY_FILTER, [])
      filter_configs = [FilterConfig(x) for x in filter_configs]
      modality_config[Constants.CONFIG_FIELD_MODALITY_FILTER] = filter_configs

      # Set default values if not specified in the config
      modality_config[Constants.CONFIG_FIELD_SAMPLE] = []
      modality_config.setdefault(
          Constants.CONFIG_FIELD_MODALITY_DIST,
          Constants.DEFAULT_MODALITY_DISTRIBUTION.get(modality_type))
      modality_config = ModalityConfig(modality_config)

      # Save the processed result to self.modality_names, self.modality and self.modality_filter
      self.modality.setdefault(modality_name, modality_config)
      self.filter.setdefault(modality_name, filter_configs)
    
    self.modality_names = list(self.modality.keys())
      
    return

  def setup_config_sample(self) -> None:
    """Setup sample related config. This function does the following:
    1. Setup config for sample, transform the list of sample config to a 
    OrderedDict where the keys are the sample names, and the values are 
    the configuration for the sample. The order of the insertion depends 
    on the order the samples appear.
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
      # Get sample information
      sample_name = sample_config.get('name', f"sample/{i:02d}")
      sample_description = sample_config.get(Constants.CONFIG_FIELD_SAMPLE_DESCRIPTION, f"sample/{i:>02}")
      sample_config['name'] = sample_name
      sample_config[Constants.CONFIG_FIELD_SAMPLE_DESCRIPTION] = sample_description

      # Check if all required keys are in the sample config.
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_SAMPLE_REQUIRED,
          sample_config,
          Constants.CONFIG_FIELD_SAMPLE,
          sample_name)
      self.sample.setdefault(sample_name, SampleConfig(sample_config))    
      
      # Check each modality config in the sample
      for i, sample_modality_config in enumerate(sample_config.get(Constants.CONFIG_FIELD_MODALITY)):
        # Check if the modality name is in the self.modality
        modality_name = sample_modality_config.get('name', '')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
        self.sample[sample_name][Constants.CONFIG_FIELD_MODALITY][i]['name'] = modality_name
        if modality_name in self.modality:
          # Put the sample names into self.modality['modality_name']
          self.modality.get(modality_name).get(Constants.CONFIG_FIELD_SAMPLE).append(sample_name)
        else:
          message = "".join((
            f"No configuration for modality {modality_name} of sample {sample_name} ",
            f"in the config file {self.filename}."))
          raise KeyError(message)
        # Check if all required keys are in the modalilty of the sample.
        self.are_all_fields_in_mapping(
            Constants.CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED,
            sample_modality_config,
            sample_name,
            Constants.CONFIG_FIELD_MODALITY)
    
    return  


  def setup_training_dataset(
      self,
      default_optimizer: str = 'adam',
      default_learning_rate: float = 1e-3,
      default_max_n_epochs: int = 150,
      default_batch_size: int = 128) -> None:
    """Setup training related config.

    Parameters
    ----------
    default_optimizer: str, optional
        default optimizer. Defaults to 'adam'.
    
    default_learning_rate: float, optional
        default learning rate for optimizer. Defaults to 5e-4.

    default_max_n_epochs: int, optional
        default maximum number of epochs for training. Defaults to 500.

    default_batch_size: int, optional
        default batch size for training and predict. Defaults to 128.

    """
    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    # Set default values if not specified in the training config
    training_config = model_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING, {})
    optimizer_config = training_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER, {})
    optimizer_config.setdefault('name', default_optimizer)
    optimizer_config.setdefault(
        Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE, default_learning_rate
    )
    optimizer_config[Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE] = float(
        optimizer_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE))
    optimizer_config = OptimizerConfig(optimizer_config)

    training_config[Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER] = optimizer_config
    training_config.setdefault(
        Constants.CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS, 
        default_max_n_epochs)
    self.training = TrainingConfig(training_config)

    # Set default values if not specified in the dataset config
    dataset_config = model_config.get(Constants.CONFIG_FIELD_MODEL_DATASET, {})
    dataset_config.setdefault(
        Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE,
        default_batch_size)
    self.dataset = DatasetConfig(dataset_config)

  def setup_config_model(
      self,
      default_n_latent_dims: int = 5,
      default_n_encoder_layers: int = 3,
      default_n_decoder_layers: int = 3,
      default_n_progressive_epochs: int = 100) -> None:
    """Setup model and component related config.

    Parameters
    ----------
    default_n_latent_dims: int, optional
        default number of latent dimensions. Defaults to 5.

    default_n_encoder_layers: int, optional
        default number of layers in the encoder backbone. Defaults to 3.

    default_n_decoder_layers: int, optional
        default number of layers in the decoder backbone. Defaults to 3.
    
    default_n_progressive_epochs: int, optional
        default number of progressive epochs. Defaults to 100.

    Raises
    ------
    KeyError
        if any of the required key is not in the provided config.

    AttributeError
        if the dependencies between components is not a directed 
        acyclic graph.

    """
    self.model = dict()
    self.components = list()
    
    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    model_name = model_config.get('name', 'CAVACHON')

    self.are_all_fields_in_mapping(
        Constants.CONFIG_FIELD_MODEL_REQUIRED,
        model_config,
        'model config')

    # Check required field and default values if not specified in the component config
    component_config_list = model_config.get(Constants.CONFIG_FIELD_MODEL_COMPONENT)
    processed_component_config_mapping = dict()
    for i, component_config in enumerate(component_config_list):
      # Check required field
      component_name = component_config.get('name', f"Component/{i:02d}")
      component_name = GeneralUtils.tensorflow_compatible_str(component_name)
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_COMPONENT_REQUIRED,
          component_config,
          component_name,
          'component config')
      
      # Set default values for conditioend_on, n_encoder_layers, n_latent_dims and n_priors for
      # each component
      component_conditioned_on_z = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
          [])
      component_conditioned_on_z = [GeneralUtils.tensorflow_compatible_str(x) for x in component_conditioned_on_z]
      component_conditioned_on_z_hat = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT,
          [])
      component_conditioned_on_z_hat = [GeneralUtils.tensorflow_compatible_str(x) for x in component_conditioned_on_z_hat]
      component_n_progressive_epochs = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS,
          default_n_progressive_epochs
      )
      component_n_encoder_layers = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_N_ENCODER_LAYERS,
          default_n_encoder_layers)
      component_n_latent_dims = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_N_LATENT_DIMS,
          default_n_latent_dims)
      component_n_latent_priors = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_N_PRIORS,
          2 * component_n_latent_dims + 1)
      component_modalities = component_config.get(Constants.CONFIG_FIELD_MODALITY)
      component_modality_names = list() 
      component_distribution_names = dict()
      component_n_decoder_layers = dict()

      # Setup modality names and decoder for the component
      for modality_config in component_modalities:
        self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_COMPONENT_MODALITIES_REQUIRED,
          modality_config,
          component_name,
          'modalities config')
        modality_name = modality_config.get('name')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
        modality_dist = self.modality.get(modality_name).get(Constants.CONFIG_FIELD_MODALITY_DIST)
        modality_n_decoder_layer = modality_config.get(
            Constants.CONFIG_FIELD_COMPONENT_N_DECODER_LAYERS,
            default_n_decoder_layers)
        component_modality_names.append(modality_name)
        component_distribution_names.setdefault(modality_name, modality_dist)
        component_n_decoder_layers.setdefault(modality_name, modality_n_decoder_layer)

      # Store the processed_component_config to component_name
      processed_component_config = dict()
      processed_component_config.setdefault('name', component_name)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
          component_conditioned_on_z)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT,
          component_conditioned_on_z_hat)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS,
          component_n_progressive_epochs)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES,
          component_modality_names)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES,
          component_distribution_names)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_N_LATENT_DIMS,
          component_n_latent_dims)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_N_PRIORS,
          component_n_latent_priors)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_N_ENCODER_LAYERS,
          component_n_encoder_layers)
      processed_component_config.setdefault(
          Constants.CONFIG_FIELD_COMPONENT_N_DECODER_LAYERS,
          component_n_decoder_layers)

      processed_component_config = ComponentConfig(processed_component_config)
      processed_component_config_mapping.setdefault(component_name, processed_component_config)
    
    # Sort the components based on the BFS order
    sorted_component_configs = GeneralUtils.order_components(processed_component_config_mapping)
    self.components = [ComponentConfig(x) for x in sorted_component_configs]
    
    self.model = ModelConfig({
      'name': model_name,
      Constants.CONFIG_FIELD_MODEL_COMPONENT: self.components,
      Constants.CONFIG_FIELD_MODEL_TRAINING: self.training,
      Constants.CONFIG_FIELD_MODEL_DATASET: self.dataset
    })