from cavachon.config.Config import Config
from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.filter.AnnDataFilterHandler import AnnDataFilterHandler
from cavachon.io.FileReader import FileReader
from cavachon.modality.Modality import Modality
from cavachon.modality.MultiModality import MultiModality
from cavachon.model.Model import Model
from cavachon.scheduler.SequentialTrainingScheduler import SequentialTrainingScheduler

from typing import List, MutableMapping, Optional

import anndata
import muon as mu
import os
import tensorflow as tf
import warnings

class Workflow():
  """Workflow

  The configured workflow to perform analysis.

  Attibutes
  ---------
  config: Config
      configuration for the workflow.

  mdata: mu.MuData
      the multi-modality data.
  
  dataloader: DataLoader
      data loader used to create tf.data.Dataset from the input data.

  model: tf.keras.Model
      generative model created and trained as configured.
  
  train_scheduler: SequentialTrainingScheduler
      sequential training sceduler for each component in the model.

  train_history: List[tf.keras.callbacks.History]
      history of training process in each step.

  outputs: MutableMapping[str, tf.Tensor]
      outputs latent representations and reconstructed data from the
      trained generative mdoel.

  """
  def __init__(self, filename: str):
    """Constructor for Workflow.

    Parameters
    ----------
    filename: str
        path to the configuration file (config.yaml)

    """
    self.config: Config = Config(filename)
    self.mdata: Optional[mu.Mudata] = None
    self.dataloader: Optional[DataLoader] = None
    self.anndata_filters: AnnDataFilterHandler = AnnDataFilterHandler.from_config(self.config)
    self.model: Optional[Model] = None
    self.train_scheduler: Optional[SequentialTrainingScheduler] = None
    self.train_history: List[tf.keras.callbacks.History] = list()
    self.outputs: MutableMapping[str, tf.Tensor] = dict()

    return

  def run(self) -> None:
    """Run the specified workflow configured in self.config"""
    self.setup_mdata()
    self.setup_dataloader()

    self.model = Model.make(component_configs=self.config.components, name=self.config.model.name)
    
    self.setup_train_scheduler()
    if self.config.model.load_weights:
      self.load_model_weights()
    if self.config.training.train:
      self.train_model()
    
    self.predict()

    return
  
  def setup_mdata(self) -> None:
    """Setup mdata and update n_vars in the component config."""
    adata_dict = Workflow.read_modalities(self.config)
    adata_dict = self.anndata_filters(adata_dict)
    self.mdata = MultiModality(adata_dict)
    self.update_config_nvars()

  @staticmethod
  def read_modalities(config: Config) -> MutableMapping[str, anndata.AnnData]:
    """Read the modality files from the configuration.

    Returns
    -------
    MutableMapping[str, anndata.AnnData]
        the keys are the names of the modality, the values are the 
        corresponding AnnData.

    """
    modalities = dict()
    for modality_name in config.modality_names:
      modality_config = config.modality[modality_name]
      h5ad = modality_config.get(Constants.CONFIG_FIELD_MODALITY_H5AD)
      if h5ad:
        adata = anndata.read_h5ad(os.path.join(config.io.datadir, h5ad))
      else:
        adata = FileReader.read_multiomics_data(config, modality_name)
      modalities.setdefault(
          modality_name,
          Modality(
              adata, 
              name=modality_name,
              modality_type=modality_config.get(Constants.CONFIG_FIELD_MODALITY_TYPE),
              distribution_name=modality_config.get(Constants.CONFIG_FIELD_MODALITY_DIST),
              batch_effect_colnames=modality_config.get(Constants.CONFIG_FIELD_MODALITY_BATCH_COLNAMES)))
  
    return modalities
  
  def filter_adata_mapping(
      self, 
      adata_mapping: MutableMapping[str, anndata.AnnData]) -> MutableMapping[str, anndata.AnnData]:
    """Filter the mapping of provided AnnData with self.anndata_filters.

    Parameters
    ----------
    adata_mapping : MutableMapping[str, anndata.AnnData]
        provided mapping of AnnData.

    Returns
    -------
    MutableMapping[str, anndata.AnnData]
        mapping of filtered AnnData.

    """
    return self.anndata_filters(adata_mapping)

  def update_config_nvars(self) -> None:
    """Update the number of variables in the config after filtering AnnData"""
    processed_component_configs = list()
    for component_config in self.config.components:
      component_vars = dict()
      for modality_name in component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES):
        component_vars.setdefault(modality_name, self.mdata[modality_name].n_vars)

      component_config[Constants.CONFIG_FIELD_COMPONENT_N_VARS] = component_vars
      processed_component_configs.append(component_config)
    
    self.config.components = processed_component_configs
    self.config.model[Constants.CONFIG_FIELD_MODEL_COMPONENT] = processed_component_configs
  
    return

  def setup_dataloader(self) -> None:
    """Setup mdata and update n_vars_batch_effect in the component config."""
    batch_size = self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
    distribution_names = dict()
    batch_effect_colnames = dict()
    for modality_name, modality_config in self.config.modality.items():
      distribution_names.setdefault(
          modality_name, 
          modality_config.get(Constants.CONFIG_FIELD_MODALITY_DIST))
      batch_effect_colnames.setdefault(
          modality_name,
          modality_config.get(Constants.CONFIG_FIELD_MODALITY_BATCH_COLNAMES))
    
    self.dataloader = DataLoader(self.mdata, batch_size, batch_effect_colnames, distribution_names)
    self.update_nvars_batch_effect()

    return
  
  def update_nvars_batch_effect(self) -> None:
    """Update the number of batch effect variables after creating DataLoader"""
    processed_component_configs = list()
    nvars = self.dataloader.n_vars_batch_effect
    for component_config in self.config.components:
      component_vars = dict()
      for modality_name in component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES):
        component_vars[modality_name] = nvars.get(modality_name)
      component_config[Constants.CONFIG_FIELD_COMPONENT_N_VARS_BATCH] = component_vars
      processed_component_configs.append(component_config)
    
    self.config.components = processed_component_configs
    self.config.model[Constants.CONFIG_FIELD_MODEL_COMPONENT] = processed_component_configs

    return
  
  def setup_train_scheduler(self) -> None:
    """Setup the training scheduler"""
    optimizer_config = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER)
    optimizer = optimizer_config.get('name')
    learning_rate = optimizer_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE)
    early_stopping = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_EARLY_STOPPING)
    self.train_scheduler = SequentialTrainingScheduler(
        self.model,
        optimizer,
        learning_rate,
        early_stopping)
      
    return
    
  def load_model_weights(self) -> None:
    """Load the pretrained model weights"""
    try:
      self.model.load_weights(
          os.path.join(f'{self.config.io.checkpointdir}', self.model.name))
    except:
      message = ''.join((
          f'Cannot load the pretrained weights in {self.config.io.checkpointdir}.'
      ))
      warnings.warn(message, RuntimeWarning)
  
    return None
  
  def train_model(self) -> None:
    """Train the model and save the weights if model.save_weights is 
    set to True.
    
    """
    batch_size = self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
    max_epochs = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS)

    # shuffle dataset if needed
    if self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_SHUFFLE):
      self.dataloader.dataset.shuffle(self.mdata.n_obs).batch(batch_size)
    else:
      train_dataset = self.dataloader.dataset.batch(batch_size)

    # train the model
    self.train_history = self.train_scheduler.fit(
        train_dataset,
        epochs=max_epochs)
    
    # save the weights if needed
    if self.config.model.save_weights: 
      self.model.save_weights(
        os.path.join(f'{self.config.io.checkpointdir}', self.model.name))

    # change the training states to False
    for component in self.model.components.values():
      component.trainable = False
      progressive_scaler = component.hierarchical_encoder.progressive_scaler
      progressive_scaler.total_iterations.assign(1.0)
      progressive_scaler.current_iteration.assign(1.0)
      self.model.compile()
    
    return
  
  def predict(self) -> None:
    """Predict generative process for self.mdata."""
    self.model.trainable = False
    self.model.compile()
    batch_size = self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
    self.outputs = self.model.predict(self.mdata, batch_size=batch_size)
    
    return