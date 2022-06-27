#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

from cavachon.distributions.IndependentNormalWrapper import IndependentNormalWrapper
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.model.Prior import Prior
from cavachon.utils.GeneralUtils import GeneralUtils
from cavachon.utils.TensorUtils import TensorUtils
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from typing import List, Union
from time import time

class Model(tf.keras.Model):
  def __init__(
      self,
      name: str,
      modality_ordered_map: ModalityOrderedMap):
    super().__init__(name=name)
    self.is_trained: bool = False
    self.n_modalities: int = 0
    
    self.setup(modality_ordered_map)

  def setup(self, modality_ordered_map: ModalityOrderedMap, force: bool = False):
    if self.is_trained and not force:
      message = ''.join((
        'Performing setup with trained model will reset the weights, ',
        'set force=True to execute anyway.'
      ))
      warnings.warn(message, RuntimeWarning)
      return
    
    self.encoder_backbone_networks = []
    self.decoder_backbone_networks = []
    self.z_parameterizers = []
    self.z_prior = []
    self.decoder_r_networks = []
    self.decoder_b_networks = []
    for name, modality in modality_ordered_map.data.items():
      n_layers = modality.n_layers
      n_clusters = modality.n_clusters
      n_latent_dims = modality.n_latent_dims
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
          IndependentNormalWrapper.export_parameterizer(
              n_latent_dims,
              name=f"{name}:z_parameterizers"))
      self.z_prior.append(Prior(name, n_latent_dims, n_clusters))
      self.decoder_r_networks.append(
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_r"))
      self.decoder_b_networks.append(
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_b"))

    
    return