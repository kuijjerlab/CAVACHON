import networkx as nx
import os
from cavachon.environment.Constants import Constants
from cavachon.io.FileReader import FileReader
from cavachon.utils.GeneralUtils import GeneralUtils
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

class Config:
  """ConfigParser
  [TODO: DEPRECATED DOCUMENTATION]
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
  
  def __init__(
      self,
      filename: str,
      default_data_dir: str = './',
      default_optimizer: str = 'adam',
      default_learning_rate: float = 1e-3,
      default_max_n_epochs: int = 150,
      default_batch_size: int = 128,
      default_n_latent_dims: int = 5,
      default_n_encoder_layers: int = 3,
      default_n_decoder_layers: int = 3) -> None:
    self.datadir: str = ''
    self.filename = os.path.realpath(filename)
    self.yaml: Dict[str, Any] = FileReader.read_yaml(filename)
    self.io: Dict[str, str] = dict()
    self.sample: OrderedDict[str, Any] = OrderedDict()
    self.modality: Dict[str, Any] = dict()
    self.modality_names: List[str] = list()
    self.modality_filter: Dict[str, Any] = dict()
    self.model: Dict[str, Any] = dict()
    self.training: Dict[str, Any] = dict()
    self.components: OrderedDict[str, Any] = OrderedDict()
    self.dataset: Dict[str, Any] = dict()
    
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
        default_n_decoder_layers=default_n_decoder_layers)

    return

  def are_all_fields_in_mapping(
      self,
      key_list: List[Any],
      mapping: Mapping,
      field: str,
      subfield: str = '') -> bool:
    """Check if all the keys are in the Mapping

    Args:
        key_list (List[Any]): the keys to be evaluated.
        
        mapping (Mapping): the mapping to be evaluated.

    Returns:
        Tuple[bool, List[Any]]: the first element is whether the all the
        keys are in the Mapping. The second element is a list of keys 
        that do not exist in the Mappipng.
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
    """Setup IO related directories"""
    self.io = self.yaml.get(Constants.CONFIG_FIELD_IO)
    datadir = self.io.get(Constants.CONFIG_FIELD_IO_DATADIR, default_datadir)
    self.datadir = os.path.realpath(os.path.dirname(f'{datadir}/'))
    return

  def setup_config_modality(self) -> None:
    """This function does the following:
    1. Setup config for modalities, transform the list of modality 
    config to a OrderedDict where the keys are the modality names, and 
    the values are the configuration for the modality. The order of the
    insertion depends on the order field in the config (in ascending
    order)
    2. Check all the relevant fields are in the config. Set defaults if 
    the optional fields are not provided.
    """
    # Clear self.modality and self.modality_filter
    self.modality = dict()
    self.modality_filter = dict()

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

      # Set default values if not specified in the config
      modality_config[Constants.CONFIG_FIELD_SAMPLE] = []
      modality_config.setdefault(
          Constants.CONFIG_FIELD_MODALITY_DIST,
          Constants.DEFAULT_MODALITY_DISTRIBUTION.get(modality_type))

      # Save the processed result to self.modality_names, self.modality and self.modality_filter
      self.modality.setdefault(modality_name, modality_config)
      self.modality_filter.setdefault(
          modality_name,
          modality_config.get(Constants.CONFIG_FIELD_MODALITY_FILTER, {}))
    
    self.modality_names = list(self.modality.keys())
      
    return

  def setup_config_sample(self) -> None:
    """This function does the following:
    1. Setup config for sample, transform the list of sample config to a 
    OrderedDict where the keys are the sample names, and the values are 
    the configuration for the sample. The order of the insertion depends 
    on the order the samples appear.
    2. Save the sample modality config to each modality in the 
    config_modality.
    3. Check all the relevant fields are in the config. Set defaults if 
    the optional fields are not provided.

    Raises:
      KeyError: raise if the configuration for the modality of any 
      sample cannot be found in config_modality.
    """
    # Clear the OrderedDict of config sample
    self.sample = OrderedDict()
    # Get the list of sample config
    sample_config_list = self.yaml.get(Constants.CONFIG_FIELD_SAMPLE, [])
    for i, sample_config in enumerate(sample_config_list):
      # Get sample information
      sample_name = sample_config.get('name', f"sample/{i:02d}")
      sample_description = sample_config.get('description', f"sample/{i:>02}")
      sample_config['name'] = sample_name
      sample_config['description'] = sample_description

      # Check if all required keys are in the sample config.
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_SAMPLE_REQUIRED,
          sample_config,
          Constants.CONFIG_FIELD_SAMPLE,
          sample_name)
      self.sample.setdefault(sample_name, sample_config)    
      
      # Check each modality config in the sample
      for sample_modality_config in sample_config.get(Constants.CONFIG_FIELD_MODALITY):
        # Check if the modality name is in the self.modality
        modality_name = sample_modality_config.get('name', '')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
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

  def setup_training_dataset(
      self,
      default_optimizer: str = 'adam',
      default_learning_rate: float = 1e-3,
      default_max_n_epochs: int = 150,
      default_batch_size: int = 128) -> None:
    self.training = dict()
    self.dataset = dict()

    model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
    # Set default values if not specified in the training config
    self.training = model_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING, {})
    optimizer_config = self.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER, {})
    optimizer_config.setdefault('name', default_optimizer)
    optimizer_config.setdefault(
        Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE, default_learning_rate
    )
    self.training[Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER] = optimizer_config
    self.training.setdefault(
        Constants.CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS, 
        default_max_n_epochs)
    
    # Set default values if not specified in the dataset config
    self.dataset = model_config.get(Constants.CONFIG_FIELD_MODEL_DATASET, {})
    self.dataset.setdefault(
        Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE,
        default_batch_size)

  def setup_config_model(
      self,
      default_n_latent_dims: int = 5,
      default_n_encoder_layers: int = 3,
      default_n_decoder_layers: int = 3) -> None:
    self.model = dict()
    self.components = OrderedDict()
    
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
      self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_MODEL_COMPONENT_REQUIRED,
          component_config,
          component_name,
          'component config')
      
      # Set default values for conditioend_on, n_encoder_layers, n_latent_dims and n_priors for
      # each component
      component_conditioned_on = component_config.get('conditioned_on', [])
      component_n_encoder_layers = component_config.get(
          Constants.CONFIG_FIELD_MODEL_COMPONENT_N_ENCODER_LAYERS,
          default_n_encoder_layers)
      component_n_latent_dims = component_config.get(
          Constants.CONFIG_FIELD_MODEL_COMPONENT_N_LATENT_DIMS,
          default_n_latent_dims)
      component_n_priors = component_config.get(
          Constants.CONFIG_FIELD_MODEL_COMPONENT_N_PRIORS,
          2 * component_n_latent_dims + 1)
      component_modalities = component_config.get(Constants.CONFIG_FIELD_MODALITY)
      component_modality_names = list() 
      component_distribution_names = dict()
      component_n_decoder_layers = dict()
      # Setup modality names and decoder for the component
      for modality_config in component_modalities:
        self.are_all_fields_in_mapping(
          Constants.CONFIG_FIELD_MODEL_COMPONENT_MODALITIES_REQUIRED,
          modality_config,
          component_name,
          'modalities config')
        modality_name = modality_config.get('name')
        modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
        modality_dist = self.modality.get(modality_name).get(Constants.CONFIG_FIELD_MODALITY_DIST)
        modality_n_decoder_layer = modality_config.get(
            Constants.CONFIG_FIELD_MODEL_COMPONENT_N_DECODER_LAYERS,
            default_n_decoder_layers)
        component_modality_names.append(modality_name)
        component_distribution_names.setdefault(modality_name, modality_dist)
        component_n_decoder_layers.setdefault(modality_name, modality_n_decoder_layer)

      # Store the processed_component_config to component_name
      processed_component_config = {
          'name': component_name,
          'conditioned_on': component_conditioned_on,
          'modality_names': component_modality_names,
          'distribution_names': component_distribution_names,
          'n_latent_dims': component_n_latent_dims,
          'n_priors': component_n_priors,
          'n_encoder_layers': component_n_encoder_layers,
          'n_decoder_layers': component_n_decoder_layers
      }
      processed_component_config_mapping.setdefault(component_name, processed_component_config)
    
    self.components = self.order_components(processed_component_config_mapping)
    self.model = {
      'name': model_name,
      'components': self.components
    }

  def order_components(
      self,
      component_config_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    component_id_mapping = dict()
    id_component_mapping = dict()
    for i, (component_name, component_config) in enumerate(component_config_mapping.items()):
      component_id_mapping.setdefault(component_name, i)
      id_component_mapping.setdefault(i, component_config)

    n_components = len(component_config_mapping)
    component_ids = list(range(n_components))

    G = nx.DiGraph()
    for i in component_ids:
      G.add_node(i)
    for component_id, component_config in id_component_mapping.items():
      conditioned_on = component_config.get('conditioned_on')
      if conditioned_on is not None or len(conditioned_on) != 0:
        for conditioned_on_component_name in conditioned_on:
          conditioned_on_component_id = component_id_mapping.get(conditioned_on_component_name)
          G.add_edge(component_id, conditioned_on_component_id)
    
    if not nx.is_directed_acyclic_graph(G):
      message = ''.join((
          f'The conditioning relationships between components form a directed cyclic graph. ',
          f'Please check the conditioning relationships between components in the config file ',
          f'{self.filename}.\n'))
      raise AttributeError(message)
    
    component_id_ordered_list = list()
    while len(component_id_ordered_list) != n_components:
      for component_id in G.nodes:
        node_successors_not_added = set()
        for bfs_successors in nx.bfs_successors(G, component_id):
          node, successors = bfs_successors
          node_successors_not_added = node_successors_not_added.union(set(successors))
        node_successors_not_added = node_successors_not_added.difference(
            set(component_id_ordered_list))
        if len(node_successors_not_added) == 0:
          component_id_ordered_list.append(component_id)

    components = [id_component_mapping[component_id] for component_id in component_id_ordered_list]
    return components