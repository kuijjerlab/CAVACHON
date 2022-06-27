#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.model.Prior import Prior
from cavachon.utils.GeneralUtils import GeneralUtils
from cavachon.utils.TensorUtils import TensorUtils
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Union
from time import time

class Model(tf.keras.Model):
  def __init__(
      self,
      name: str,
      modality_ordered_map: ModalityOrderedMap):
    super().__init__(name=name)
    self.is_trained: bool = False
    self.n_modalities: int = 0
    self.modality_names: List[str] = []
    self.encoder_backbone_networks: List[tf.keras.Model] = []
    self.decoder_backbone_networks: List[tf.keras.Model] = []
    self.z_parameterizers: List[Dict[str, tf.keras.Model]] = []
    self.decoder_r_networks: List[tf.keras.Model] = []
    self.decoder_b_networks: List[tf.keras.Model] = []
    self.x_parameterizers: List[Dict[str, tf.keras.Model]] = []

    self.setup(modality_ordered_map)

  def call(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Tuple[List[Dict[str, tf.Tensor]], List[tf.Tensor]]:

    z_parameters = self.encode(inputs, training=training)
    z_list, z_hat_list = self.encode_across_modalities(z_parameters, training=training)
    x_parameters = self.decode(inputs, z_hat_list, training=training)

    return z_parameters, z_list, z_hat_list, x_parameters

  def encode(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False) -> List[Dict[str, tf.Tensor]]:
    
    z_parameters = []
    for i, modality_name in enumerate(self.modality_names):
      x_encoded = self.encoder_backbone_networks[i](
          tf.sparse.to_dense(inputs.get(f'{modality_name}:matrix')),
          training=training)
      parameters = dict()
      for parameter_name, parameterizer in self.z_parameterizers[i].items():
        parameters.setdefault(parameter_name, parameterizer(x_encoded, training=training))
      z_parameters.append(parameters)

    return z_parameters
    
  def encode_across_modalities(
      self,
      z_parameters: List[Dict[str, tf.Tensor]],
      training: bool = False) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    
    z_list = [None] * self.n_modalities
    z_hat_list = [None] * self.n_modalities
    
    for index in reversed(range(0, self.n_modalities)):
      # eq D.63 (Falck et al., 2021)
      if training:
        z = MultivariateNormalDiagWrapper(
          z_parameters[index].get('mean'),
          z_parameters[index].get('var')).dist.sample()
      else:
        z = z_parameters[index].get('mean')
      
      if index == self.n_modalities - 1:
        # eq D.64 (Falck et al., 2021)
        z_hat = self.decoder_r_networks[index](z, training=training)
        z_hat = self.decoder_b_networks[index](z_hat, training=training)
      else:
        # eq D.65 (Falck et al., 2021)
        z_hat = self.decoder_r_networks[index](z, training=training)
        z_hat = tf.concat([z_hat_list[index + 1], z_hat], axis=1)
        z_hat = self.decoder_b_networks[index](z_hat, training=training)
      
      z_list[index] = z
      z_hat_list[index] = z_hat

    return z_list, z_hat_list

  def decode(
      self,
      inputs: Dict[str, tf.Tensor],
      z_hat_list: List[tf.Tensor],
      training: bool = False):
    
    x_parameters = []
    for i, modality_name in enumerate(self.modality_names):
      batch_effect = inputs.get(f'{modality_name}:batch_effect')
      decoder_input = tf.concat([z_hat_list[i], batch_effect], axis=1)
      z_decoded = self.decoder_backbone_networks[i](
          decoder_input,
          training=training)
      parameters = dict()
      for parameter_name, parameterizer in self.x_parameterizers[i].items():
        parameters.setdefault(parameter_name, parameterizer(z_decoded, training=training))
      x_parameters.append(parameters)
    
    return x_parameters

  def setup(self, modality_ordered_map: ModalityOrderedMap, force: bool = False) -> None:
    if self.is_trained and not force:
      message = ''.join((
        'Performing setup with trained model will reset the weights, ',
        'set force=True to execute anyway.'
      ))
      warnings.warn(message, RuntimeWarning)
      return
    
    self.n_modalities = len(modality_ordered_map.data)
    self.modality_names = []
    self.encoder_backbone_networks = []
    self.decoder_backbone_networks = []
    self.z_parameterizers = []
    self.z_prior = []
    self.decoder_r_networks = []
    self.decoder_b_networks = []
    self.x_parameterizers = []
    for name, modality in modality_ordered_map.data.items():
      self.modality_names.append(name)
      n_layers = modality.n_layers
      n_clusters = modality.n_clusters
      n_latent_dims = modality.n_latent_dims
      n_data_dims = modality.adata.n_vars
      self.encoder_backbone_networks.append(
          TensorUtils.create_backbone_layers(
              n_layers,
              name=f"{name}:encoder_backbone"))
      self.decoder_backbone_networks.append(
          TensorUtils.create_backbone_layers(
              n_layers,
              reverse=True,
              name=f"{name}:decoder_backbone"))
      self.z_parameterizers.append(
          MultivariateNormalDiagWrapper.export_parameterizer(
              n_latent_dims,
              name=f"{name}:z_parameterizer"))
      self.z_prior.append(Prior(name, n_latent_dims, n_clusters))
      self.decoder_r_networks.append(
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_r"))
      self.decoder_b_networks.append(
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_b"))
      self.x_parameterizers.append(
          modality.dist_cls.export_parameterizer(n_data_dims, name=f"{name}:x_parameterizer")
      )

    return
# %%

