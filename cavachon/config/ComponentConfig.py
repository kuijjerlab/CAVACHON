from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, List, Mapping

class ComponentConfig(ConfigMapping):
  """ComponentConfig

  Config for component.

  Attributes
  ----------
  name: str
      name of the component.
  
  conditioned_on: List[str]
      names of the conditioned components.
  
  modality_names: List[str]
      names of modalities used in inputs and outputs.

  distribution_names: Mapping[str, str]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are the corresponding 
      distribution names.
  
  n_vars: Mapping[str, int]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are the number of 
      variables.

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
    self.conditioned_on: List[str] = list()
    self.modality_names: List[str] = list()
    self.distribution_names: Mapping[str, str] = dict()
    self.n_vars: Mapping[str, int] = dict()
    self.n_latent_dims: int
    self.n_latent_priors: int
    self.n_encoder_layers: int
    self.n_decoder_layers: Mapping[str, int] = dict()
    self.n_progressive_epochs: int
    super().__init__(config)