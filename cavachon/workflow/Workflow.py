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
import os
import tensorflow as tf

class Workflow():
  """Workflow

  The configured workflow to perform analysis.

  Attibutes
  ---------
  config: Config
      configuration for the workflow.
  
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
    self.dataloader: Optional[DataLoader] = None
    self.anndata_filters: AnnDataFilterHandler = AnnDataFilterHandler.from_config(self.config)
    self.model: Optional[Model] = None
    self.train_scheduler: Optional[SequentialTrainingScheduler] = None
    self.train_history: List[tf.keras.callbacks.History] = list()
    self.outputs: MutableMapping[str, tf.Tensor] = dict()

    optimizer_config = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER)
    learning_rate = float(optimizer_config.get(
        Constants.CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE , 
        5e-4))
    self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.get(
        optimizer_config.get('name', 'adam')).__class__(learning_rate=learning_rate)

    return

  def run(self) -> None:
    """Run the specified workflow configured in self.config"""
    self.read_modalities()
    self.filter_anndata()
    self.dataloader = DataLoader(self.multi_modalities)
    self.update_nvars_batch_effect()
    self.model = Model.make(self.config.components)
    self.train_scheduler = SequentialTrainingScheduler(self.model, self.optimizer)
    batch_size = self.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
    max_epochs = self.config.training.get(Constants.CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS)
    self.train_history = self.train_scheduler.fit(
        self.dataloader.dataset.batch(batch_size),
        epochs=max_epochs)
    self.outputs = self.model.predict(self.multi_modalities, batch_size=batch_size)

    return
  
  def read_modalities(self) -> None:
    """Read the modality files and stored the data in self.modalities"""
    self.modalities = dict()
    for modality_name in self.config.modality_names:
      modality_config = self.config.modality[modality_name]
      h5ad = modality_config.get(Constants.CONFIG_FIELD_MODALITY_H5AD)
      if h5ad:
        self.modalities.setdefault(
            modality_name,
            anndata.read_h5ad(os.path.join(self.config.io.datadir, h5ad)))
      else:
        self.modalities.setdefault(
            modality_name,
            Modality(
                FileReader.read_multiomics_data(self.config, modality_name), 
                name=modality_name,
                distribution_name=modality_config.get(Constants.CONFIG_FIELD_MODALITY_DIST),
                modality_type=modality_config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)))
    return

  def filter_anndata(self) -> None:
    """Filter the AnnData as configured in self.config.filter"""
    self.anndata_filters(self.modalities)
    self.multi_modalities = MultiModality(self.modalities)
    self.update_config_nvars()

    return
  
  def update_config_nvars(self) -> None:
    """Update the number of variables after filtering AnnData"""
    processed_component_configs = list()
    for component_config in self.config.components:
      component_vars = dict()
      for modality_name in component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES):
        modality = self.modalities.get(modality_name)
        component_vars.setdefault(modality_name, modality.n_vars)
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