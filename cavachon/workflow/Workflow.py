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
    adata_dict = Workflow.read_modalities(self.config)
    self.anndata_filters(adata_dict)
    self.mdata = MultiModality(adata_dict)
    self.update_config_nvars()

    self.dataloader = DataLoader(self.mdata)
    self.update_nvars_batch_effect()

    self.model = Model.make(
        component_configs=self.config.components,
        name=self.config.model.name)
    
    optimizer_config = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER)
    optimizer = optimizer_config.get('name', 'adam')
    learning_rate = float(optimizer_config.get(
        Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE, 
        1e-3))
    early_stopping = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_EARLY_STOPPING)
    self.train_scheduler = SequentialTrainingScheduler(
        self.model,
        optimizer,
        learning_rate,
        early_stopping)
    
    batch_size = self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
    max_epochs = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS)
    if self.config.model.load_weights:
      try:
        self.model.load_weights(
            os.path.join(f'{self.config.io.checkpointdir}', self.model.name))
      except:
        self.config.training[Constants.CONFIG_FIELD_MODEL_TRAINING_TRAIN] = True
        message = ''.join((
            f'Cannot load the pretrained weights in {self.config.io.checkpointdir}, force retrain '
            f'the model.'
        ))
        warnings.warn(message, RuntimeWarning)
        
    if self.config.training.train:
      if self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_SHUFFLE):
        self.dataloader.dataset.shuffle(self.mdata.n_obs, reshuffle_each_iteration=False).batch(batch_size)
      else:
        train_dataset = self.dataloader.dataset.batch(batch_size)
      self.train_history = self.train_scheduler.fit(
          train_dataset,
          epochs=max_epochs)
      if self.config.model.save_weights: 
        self.model.save_weights(
          os.path.join(f'{self.config.io.checkpointdir}', self.model.name))
    for component in self.model.components.values():
      component.trainable = False
      progressive_scaler = component.hierarchical_encoder.progressive_scaler
      progressive_scaler.total_iterations.assign(1.0)
      progressive_scaler.current_iteration.assign(1.0)
      self.model.compile()
    self.outputs = self.model.predict(self.mdata, batch_size=batch_size)

    return
  
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
        modalities.setdefault(
            modality_name,
            anndata.read_h5ad(os.path.join(config.io.datadir, h5ad)))
      else:
        modalities.setdefault(
            modality_name,
            Modality(
                FileReader.read_multiomics_data(config, modality_name), 
                name=modality_name,
                distribution_name=modality_config.get(Constants.CONFIG_FIELD_MODALITY_DIST),
                modality_type=modality_config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)))
    
    return modalities
  
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