#%%
from __future__ import annotations
from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.model.Module import Module
from cavachon.utils.GeneralUtils import GeneralUtils
from cavachon.utils.TensorUtils import TensorUtils
from collections import OrderedDict
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Union
from time import time

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

class Model(tf.keras.Model):
  def __init__(
      self,
      name: str,
      modality_ordered_map: ModalityOrderedMap):
    super().__init__(name=name)
    self.is_trained: bool = False
    self.n_modalities: int = len(modality_ordered_map.data)
    self.modality_names: List[str] = list(modality_ordered_map.data)
    self.module = Module(modality_ordered_map)

  def call(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Tuple[List[Dict[str, tf.Tensor]], List[tf.Tensor]]:

    z_parameters = self.encode(inputs, training=training)
    z, z_hat = self.encode_across_modalities(z_parameters, training=training)
    x_parameters = self.decode(inputs, z_hat, training=training)

    return z_parameters, z, z_hat, x_parameters

  def encode(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False) -> OrderedDict[str, Dict[str, tf.Tensor]]:
    
    z_parameters = OrderedDict()
    for modality_name in self.modality_names:
      network = self.module.encoder_backbone_networks.get(modality_name)
      x_encoded = network(
          tf.sparse.to_dense(inputs.get(f'{modality_name}:matrix')),
          training=training)

      parameters = dict()
      for parameter_name, parameterizer in self.module.z_parameterizers.get(modality_name).items():
        parameters.setdefault(parameter_name, parameterizer(x_encoded, training=training))
      z_parameters.setdefault(modality_name, parameters)

    return z_parameters
    
  def encode_across_modalities(
      self,
      z_parameters: OrderedDict[str, Dict[str, tf.Tensor]],
      training: bool = False) -> Tuple[OrderedDict[str, tf.Tensor], OrderedDict[str, tf.Tensor]]:
    
    z_list = [None] * self.n_modalities
    z_hat_list = [None] * self.n_modalities
    
    for index in reversed(range(0, self.n_modalities)):
      modality_name = self.modality_names[index]
      # eq D.63 (Falck et al., 2021)
      if training:
        z = MultivariateNormalDiagWrapper(
          z_parameters.get(modality_name).get('mean'),
          z_parameters.get(modality_name).get('var')).dist.sample()
      else:
        z = z_parameters.get(modality_name).get('mean')
      
      if index == self.n_modalities - 1:
        # eq D.64 (Falck et al., 2021)
        z_hat = self.module.decoder_r_networks.get(modality_name)(z, training=training)
        z_hat = self.module.decoder_b_networks.get(modality_name)(z_hat, training=training)
      else:
        # eq D.65 (Falck et al., 2021)
        z_hat = self.module.decoder_r_networks.get(modality_name)(z, training=training)
        z_hat = tf.concat([z_hat_list[index + 1], z_hat], axis=1)
        z_hat = self.module.decoder_b_networks.get(modality_name)(z_hat, training=training)
      
      z_list[index] = z
      z_hat_list[index] = z_hat

    z_map = OrderedDict()
    z_hat_map = OrderedDict()
    for i, modality_name in enumerate(self.modality_names):
      z_map.setdefault(modality_name, z_list[i])
      z_hat_map.setdefault(modality_name, z_hat_list[i])

    return z_map, z_hat_map

  def decode(
      self,
      inputs: Dict[str, tf.Tensor],
      z_hat: OrderedDict[str, tf.Tensor],
      training: bool = False) -> OrderedDict[str, Dict[str, tf.Tensor]]:
    
    x_parameters = OrderedDict()
    for modality_name in self.modality_names:
      batch_effect = inputs.get(f'{modality_name}:batch_effect')
      decoder_input = tf.concat([z_hat.get(modality_name), batch_effect], axis=1)
      z_decoded = self.module.decoder_backbone_networks.get(modality_name)(
          decoder_input,
          training=training)
      parameters = dict()
      for parameter_name, parameterizer in self.module.x_parameterizers.get(modality_name).items():
        parameters.setdefault(parameter_name, parameterizer(z_decoded, training=training))
      x_parameters.setdefault(modality_name, parameters)
    
    return x_parameters

  def train_step(self, data):
    with tf.GradientTape() as tape:
      result = self(data, training = True)
      loss = self.compiled_loss(result, result)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(result, result)
    return { metric.name: metric.result() for metric in self.metrics }