from cavachon.config.Config import Config
from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.filter.AnnDataFilterHandler import AnnDataFilterHandler
from cavachon.io.FileReader import FileReader
from cavachon.modality.Modality import Modality
from cavachon.modality.MultiModality import MultiModality
from cavachon.model.Model import Model

from typing import Optional

import tensorflow as tf

class Workflow():
  def __init__(self, filename: str):
    self.config: Config = Config(filename)
    self.data_loader: Optional[DataLoader] = None
    self.anndata_filters: AnnDataFilterHandler = AnnDataFilterHandler.from_config(self.config)
    self.model: Optional[Model] = None
    
    optimizer_config = self.config.training.get('optimizer')
    learning_rate = float(optimizer_config.get('learning_rate', 5e-4))
    self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.get(
        optimizer_config.get('name', 'adam')).__class__(learning_rate=learning_rate)

    return

  def run(self) -> None:
    self.read_modalities()
    self.filter_anndata()
    self.data_loader = DataLoader(self.multi_modalities)
    self.model = Model.make(self.config.components)
    self.model.compile(optimizer=self.optimizer)
    self.model.fit(
        self.data_loader.dataset.batch(self.config.dataset.get('batch_size', 256)),
        epochs=self.config.training.get('max_n_epochs', 500))

    return
  
  def read_modalities(self) -> None:
    self.modalities = dict()
    for modality_name in self.config.modality_names:
      modality_config = self.config.modality[modality_name]
      self.modalities.setdefault(
          modality_name,
          Modality(
              FileReader.read_multiomics_data(self.config, modality_name), 
              name=modality_name,
              distribution_name=modality_config.get(Constants.CONFIG_FIELD_MODALITY_DIST),
              modality_type=modality_config.get(Constants.CONFIG_FIELD_MODALITY_TYPE)))
    return

  def filter_anndata(self) -> None:
    self.anndata_filters(self.modalities)
    self.multi_modalities = MultiModality(self.modalities)
    self.update_config_nvars()

    return
  
  def update_config_nvars(self) -> None:
    processed_component_configs = list()
    for component_config in self.config.components:
      component_vars = dict()
      for modality_name in component_config.get('modality_names'):
        modality = self.modalities.get(modality_name)
        component_vars.setdefault(modality_name, modality.n_vars)
      component_config['n_vars'] = component_vars
      processed_component_configs.append(component_config)
    
    self.config.components = processed_component_configs
    self.config.model['components'] = processed_component_configs
  
    return