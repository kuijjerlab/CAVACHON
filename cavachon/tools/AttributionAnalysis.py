from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.tools.DifferentialAnalysis import DifferentialAnalysis
from cavachon.utils.ReflectionHandler import ReflectionHandler
from cavachon.utils.TensorUtils import TensorUtils
from copy import deepcopy
from typing import Set, Sequence, List, Optional, Union
from tqdm import tqdm

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

class AttributionAnalysis:
  """AttributionAnalysis

  Attribution analysis of the latent representation of the component to 
  the outputs.

  Attibutes
  ---------
  mdata: muon.MuData
      the MuData for analysis.

  model: tf.keras.Model
      the trained generative model.

  """
  def __init__(
      self,
      mdata: mu.MuData,
      model: tf.keras.Model):
    """Constructor for ContributionAnalysis.

    Parameters
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.
    
    """
    self.mdata = mdata
    self.model = model

  def compute_delta_x(
      self,
      component: str,
      modality: str,
      exclude_components: List[str],
      batch_size: int = 128
  ) -> np.ndarray:
    """Compute the x - x_baseline in the integrated gradients. The 
    baseline is the mean of the outputs modality from the component
    without using the latent representation z of the exclude component.

    Parameters
    ----------
    component : str
        the outputs of which component to used.
    
    modality : str
        which modality of the outputs of the component to used.
    
    exclude_components : List[str]
        which component to exclude (the latent representation z will 
        not be used in the forward pass)
    
    batch_size : int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        x - x_baseline in the integrated gradients.

    """
    analysis = DifferentialAnalysis(mdata=self.mdata, model=self.model)
    
    for exclude_component in exclude_components:
      exclude_component_network = self.model.components[exclude_component]
      exclude_component_network.set_progressive_scaler_iteration(0, 1)
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    x_means_null = analysis.compute_x_means(
          dataset=dataloader.dataset,
          component=component,
          modality=modality,
          training=False,
          batch_size=batch_size)
      
    for exclude_component in exclude_components:
      exclude_component_network = self.model.components[exclude_component]
      exclude_component_network.set_progressive_scaler_iteration(1, 1)
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    x_means_full = analysis.compute_x_means(
          dataset=dataloader.dataset,
          component=component,
          modality=modality,
          training=False,
          batch_size=batch_size)

    return np.reshape(np.mean(np.abs(x_means_full - x_means_null), axis=-1), (-1, 1))
    
  def compute_integrated_gradient(
      self,
      component: str,
      modality: str,
      target_component: str, 
      steps: int = 10,
      batch_size: int = 128) -> np.ndarray:
    """Compute the integrated gradients of ∂rho_m/∂z_m.

    Parameters
    ----------
    component: str
        the outputs of which component to used.

    modality: str
        which modality of the outputs of the component to used.

    target_component: str
        the latent representation of which component to used.

    steps: int, optional
        steps in integrated gradients. Defaults to 10.

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        integrated gradients of ∂rho_m/∂z_m.

    """

    component_network = self.model.components[component]
    integrated_gradients = []
    
    delta_x = self.compute_delta_x(
        component = component,
        modality = modality,
        exclude_components = [target_component],
        training = False,
        batch_size = batch_size)
    
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    delta_x_split = TensorUtils.split(delta_x, batch_size=batch_size)

    for batch, batch_delta_x in tqdm(zip(dataloader, delta_x_split)):
      outputs = self.model(batch, training=False)
      
      unintegrated_gradients_batch = [] 
      for alpha in [x / steps for x in range(steps + 1)]:
        with tf.GradientTape() as tape:
          target_variable = tf.Variable(
              outputs.get(f"{target_component}/{Constants.MODEL_OUTPUTS_Z}"))
          n_vars = target_variable.shape[-1]
          hierarchical_encoder_inputs = dict()
          hierarchical_encoder_inputs.setdefault(
              Constants.MODEL_OUTPUTS_Z,
              target_variable)

          cond_on_keys = [
            Constants.MODEL_OUTPUTS_Z,
            Constants.MODEL_OUTPUTS_Z_HAT
          ]
          cond_on_inputs_keys = [
            Constants.MODULE_INPUTS_CONDITIONED_Z,
            Constants.MODULE_INPUTS_CONDITIONED_Z_HAT
          ]
          cond_on_components = [
            component_network.conditioned_on_z,
            component_network.conditioned_on_z_hat
          ]
          for cond_on_key, cond_on_inputs_key, cond_on_component_list in zip(cond_on_keys, cond_on_inputs_keys, cond_on_components):
            cond_on_tensor = list()
            for cond_on_component in cond_on_component_list:
              cond_on_tensor.append(outputs.get(f'{cond_on_component}/{cond_on_key}'))
            if len(cond_on_tensor) != 0:
              hierarchical_encoder_inputs.setdefault(
                  cond_on_inputs_key,
                  tf.concat(cond_on_tensor, axis=-1))

          z_hat = alpha * component_network.hierarchical_encoder(hierarchical_encoder_inputs)

          modality_batch_key = f'{modality}/{Constants.TENSOR_NAME_BATCH}' 
          decoder_outputs = component_network.decoders.get(modality).backbone_network(tf.concat([z_hat, batch.get(modality_batch_key)], axis=-1))
          x_means = tf.reduce_sum(decoder_outputs ** 2, axis=-1)
          gradients = tape.gradient(x_means, target_variable)
          unintegrated_gradients_batch.append(1 / (steps + 1) * batch_delta_x * gradients)
        
      unintegrated_gradients_batch = tf.stack(unintegrated_gradients_batch, axis=-1)
      integrated_gradients_batch = tf.reduce_sum(unintegrated_gradients_batch, axis=-1)
      
      integrated_gradients.append(tf.reshape(integrated_gradients_batch, (-1, n_vars)))

    integrated_gradients = tf.concat(integrated_gradients, 0)
    
    return integrated_gradients