from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, List, Mapping

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

  n_latent_dims: int, optional
      number of latent dimensions. Defaults to 5.
  
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
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for ComponentConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        component config in mapping format.
    
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
    self.n_latent_dims: int
    self.n_latent_priors: int
    self.n_encoder_layers: int
    self.n_decoder_layers: Mapping[str, int] = dict()
    self.n_progressive_epochs: int
    super().__init__(config)