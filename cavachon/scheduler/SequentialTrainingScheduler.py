from cavachon.environment.Constants import Constants
from collections import defaultdict
from typing import List, Mapping

import itertools
import tensorflow as tf

class SequentialTrainingScheduler:
  def __init__(
      self,
      model: tf.keras.Model,
      optimizer: tf.keras.optimizers.Optimizer):
    self.model = model
    self.component_configs = self.model.component_configs
    self.optimizer = optimizer
    self.training_order = self.compute_component_training_order()
    self.feature_weight = self.compute_feature_weight()

  def compute_component_training_order(self) -> Mapping[int, List[str]]:
    training_order = list()
    component_order = dict()
    component_configs = self.component_configs
    training_order.append([])
    for component_config in component_configs:
      component_name = component_config.get('name')
      conditioned_on_z = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z, [])
      conditioned_on_z_hat = component_config.get(
          Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT, [])
      conditioned_on = itertools.chain(conditioned_on_z, conditioned_on_z_hat)
      order = max([0] + [component_order[x] + 1 for x in conditioned_on])
      if order > len(training_order) - 1:
        training_order.append([])
      training_order[order].append(component_name)
      component_order.setdefault(component_name, order)

    return training_order

  def compute_feature_weight(
      self,
      constant: bool = False) -> Mapping[str, Mapping[str, int]]:
    feature_weight_by_component = defaultdict(dict)
    for component_config in self.component_configs:
      component_name = component_config.get('name')
      feature_weight_by_modality = dict()
      if not constant:
        n_vars = component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_VARS)
        total_vars = 0
        total_scaled_weight = 0
        for modality_name, n_var in n_vars.items():
          total_vars += n_var
        for modality_name, n_var in n_vars.items():
          scaled_weight = total_vars / n_var
          total_scaled_weight += scaled_weight
          feature_weight_by_modality.setdefault(modality_name, scaled_weight)
        for modality_name, scaled_weight in feature_weight_by_modality.items():
          feature_weight_by_modality[modality_name] = scaled_weight / total_scaled_weight
      else:
        for modality_name in n_vars.keys():
          feature_weight_by_modality.setdefault(modality_name, 1.0)
      feature_weight_by_component.setdefault(component_name, feature_weight_by_modality)

    return feature_weight_by_component

  def fit(self, x: tf.data.Dataset, **kwargs):
    
    n_batches = len(x)
    learning_rate = self.optimizer.learning_rate
    history = []

    for component_order, train_components in enumerate(self.training_order):
      loss_weights = dict()
      for component_config in self.component_configs:
        component_name = component_config.get('name')
        component = self.model.components.get(component_name)
        progressive_scaler = component.hierarchical_encoder.progressive_scaler
        if component_name in train_components:
          trainable = True
          progressive_iterations = n_batches * float(
              component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS))
          #print(component_order, component_name, progressive_iterations)
          progressive_scaler.total_iterations.assign(progressive_iterations)
          progressive_scaler.current_iteration.assign(1.0)
        else:
          trainable = False
          progressive_scaler.total_iterations.assign(1.0)
          progressive_scaler.current_iteration.assign(1.0)

        component.trainable = trainable   
        loss_weights.setdefault(
            f"{component_name}/{Constants.MODEL_LOSS_KL_POSTFIX}",
            #float(trainable))
            1.0)
        for modality_name in component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES):
          loss_weights.setdefault(
              f"{component_name}/{modality_name}/{Constants.MODEL_LOSS_DATA_POSTFIX}", 
              #float(trainable) * self.feature_weight.get(component_name).get(modality_name))
              self.feature_weight.get(component_name).get(modality_name))
      
      self.model.compile(
          optimizer=self.optimizer.__class__(learning_rate=learning_rate),
          loss_weights=loss_weights)
      early_stopping = tf.keras.callbacks.EarlyStopping(
          monitor='loss',
          min_delta = 5,
          patience = 100,
          restore_best_weights=True,
          verbose=1,
      )
      history.append(self.model.fit(x, callbacks=[early_stopping], **kwargs))
    
    for component in self.model.components.values():
      component.trainable = True

    return history