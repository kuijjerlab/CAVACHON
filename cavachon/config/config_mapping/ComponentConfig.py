from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.utils.GeneralUtils import GeneralUtils
from typing import Any, List, Mapping

import warnings

class ComponentConfig(ConfigMapping):
  """ComponentConfig

  Config for component.

  Attributes
  ----------
  name: str
      name of the component.
  
  conditioned_on_z: List[str]
      names of the conditioned components (of z).
  
  conditioned_on_z_hat: List[str]
      names of the conditioned components (of z_hat).

  modality_names: List[str]
      names of modalities used in inputs and outputs.

  distribution_names: Mapping[str, str]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are the corresponding 
      distribution names.
  
  save_x: Mapping[str, bool]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are wheather or not the
      predicted x_parameters is save to the obsm of modality (with key 
      'x_parameters_`name`').  Note that `x_parameters` will not be 
      predicted by defaults if none of the modalities in the component 
      set `save_x`.

  save_z: Mapping[str, bool]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are wheather or not the
      predicted z is save to the obsm of modality (with key 
      'z_`name`').

  n_vars: Mapping[str, int]
      number of variables for the inputs data distribution. It should 
      be the size of last dimensions of inputs Tensor. The keys are 
      the modality names, and the values are the corresponding number
      of variables.
    
  n_vars_batch_effect: Mapping[str, int]
      number of variables for the batch effect tensor. It should 
      be the size of last dimensions of batch effect Tensor. The keys 
      are the modality names, and the values are the corresponding 
      number of variables.

  n_latent_dims: int
      number of latent dimensions
  
  n_latent_priors: int
      number of priors for the latent distributions.

  n_encoder_layers: int
      number of encoder layers.

  n_decoder_layers: Mapping[str, int]
      number of decoder layers for each modality. The keys are the 
      modality names, and the values are the corresponding number of 
      decoder layers.
  
  n_progressive_epochs: int
      number of progressive epochs.
  
  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for ComponentConfig. 

    Parameters
    ----------
    name: str
        name of the component.
    
    modalities: List[Mapping[str, Any]]
        mappings configured the modalities. Each elemeet inside the 
        provided list should be a mapping, where the data structure 
        should be:
        1. name: str
        2. distribution_names: str
        3. n_decoder_layers: int, optional (defaults to 3)
        4. save_x: bool, optional (defaults to True)
        5. save_z: bool, optional (defaults to True)

    conditioned_on_z: List[str], optional 
        names of the conditioned components (of z). Defaults to [].
    
    conditioned_on_z_hat: List[str], optional
        names of the conditioned components (of z_hat). Defaults to [].

    n_latent_dims: int, optional
        number of latent dimensions. Defaults to 5.
    
    n_latent_priors: int, optional
        number of priors for the latent distributions. Defaults to 11.

    n_encoder_layers: int, optional
        number of encoder layers. Defaults to 3.
    
    n_progressive_epochs: int, optional
        number of progressive epochs. Defaults to 1.
    """
    self.name: str
    self.conditioned_on_z: List[str] = list()
    self.conditioned_on_z_hat: List[str] = list()
    self.modality_names: List[str] = list()
    self.distribution_names: Mapping[str, str] = dict()
    self.save_x: Mapping[str, bool] = dict()
    self.save_z: Mapping[str, bool] = dict()
    self.n_vars: Mapping[str, int] = dict()
    self.n_vars_batch_effect: Mapping[str, int] = dict()
    self.n_latent_dims: int = 5
    self.n_latent_priors: int = 11
    self.n_encoder_layers: int = 3
    self.n_decoder_layers: Mapping[str, int] = dict()
    self.n_progressive_epochs: int = 1

    super().__init__(
        kwargs,
        [
          'name',
          'modalities',
          'conditioned_on_z',
          'conditioned_on_z_hat',
          'modality_names',
          'distribution_names',
          'save_x',
          'save_z',
          'n_vars',
          'n_vars_batch_effect',
          'n_latent_dims',
          'n_latent_priors',
          'n_encoder_layers',
          'n_decoder_layers',
          'n_progressive_epochs'
        ])

    # postprocessing
    ## name
    self.name = GeneralUtils.tensorflow_compatible_str(self.name)

    ## distribution_names, n_decoder_layers, save_x, save_z
    expected_fields = {'name', 'distribution_names', 'n_decoder_layers', 'save_x', 'save_z'}
    for modality_config in kwargs.get('modalities'):
      modality_name = modality_config.get('name')
      modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
      self.modality_names.append(modality_name)
      self.distribution_names.setdefault(
          modality_name,
          modality_config.get('distribution_names'))
      self.n_decoder_layers.setdefault(
          modality_name,
          modality_config.get('n_decoder_layers', 3))
      self.save_x.setdefault(
          modality_name,
          modality_config.get('save_x', True))
      self.save_z.setdefault(
          modality_name,
          modality_config.get('save_z', True))

      for field in set(modality_config.keys()) - expected_fields:
        message = ''.join((
          f'Unexpected field {field} in modalities of {self.name} config. ',
          f'Please check if there is any unintentional typo. Some fields ',
          f'might be set to default unexpectedly. Expected fields: {expected_fields}'
        ))
        warnings.warn(message, RuntimeWarning)