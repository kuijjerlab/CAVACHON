
from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.model.Parameterizer import Parameterizer
from cavachon.model.Prior import Prior
from cavachon.utils.TensorUtils import TensorUtils
from collections import OrderedDict
from typing import Dict
import tensorflow as tf

class Module(tf.keras.Model):
  def __init__(self, modality_ordered_map: ModalityOrderedMap):
    super().__init__()
    self.encoder_backbone_networks: OrderedDict[tf.keras.Model] = OrderedDict()
    self.decoder_backbone_networks: OrderedDict[tf.keras.Model] = OrderedDict()
    self.z_parameterizers: Parameterizer = Parameterizer()
    self.z_prior: OrderedDict[str, Prior] =  OrderedDict()
    self.decoder_r_networks: OrderedDict[tf.keras.Model] = OrderedDict()
    self.decoder_b_networks: OrderedDict[tf.keras.Model] = OrderedDict()
    self.x_parameterizers: Parameterizer = Parameterizer()

    for name, modality in modality_ordered_map.data.items():
      n_layers = modality.n_layers
      n_clusters = modality.n_clusters
      n_latent_dims = modality.n_latent_dims
      n_data_dims = modality.adata.n_vars

      self.encoder_backbone_networks.setdefault(
          name, 
          TensorUtils.create_backbone_layers(
              n_layers,
              name=f"{name}:encoder_backbone"))
      self.decoder_backbone_networks.setdefault(
          name,
          TensorUtils.create_backbone_layers(
              n_layers,
              reverse=True,
              name=f"{name}:decoder_backbone"))
      self.z_parameterizers.setdefault(
          name,
          MultivariateNormalDiagWrapper.export_parameterizer(
              n_latent_dims,
              name=f"{name}:z_parameterizer"))
      self.z_prior.setdefault(
          name,
          Prior(name, n_latent_dims, n_clusters))
      self.decoder_r_networks.setdefault(
          name,
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_r"))
      self.decoder_b_networks.setdefault(
          name,
          tf.keras.Sequential([tf.keras.layers.Dense(32)], name=f"{name}:decoder_b"))
      self.x_parameterizers.setdefault(
          name,
          modality.dist_cls.export_parameterizer(n_data_dims, name=f"{name}:x_parameterizer")
      )
    
