from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from collections.abc import Callable
from typing import Dict, Sequence, Union
from tqdm import tqdm

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

class DifferentialAnalysis(Callable):
  """DifferentialAnalysis

  Differential analysis between two groups of samples.

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
    """Constructor for DifferentialAnalysis.

    Parameters
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.
    
    """
    self.mdata = mdata
    self.model = model

  def __call__(
      self, 
      group_a_index: Union[pd.Index, Sequence[str]],
      group_b_index: Union[pd.Index, Sequence[str]],
      component: str,
      modality: str,
      z_sampling_size: int = 100,
      x_sampling_size: int = 1000,
      batch_size: int = 128):
    """Perform the differential analysis between two groups.

    Parameters
    ----------
    group_a_index : Union[pd.Index, Sequence[str]]
        index of group one. Needs to meet the index in the obs of the
        modality.
    
    group_b_index : Union[pd.Index, Sequence[str]]
        index of group two. Needs to meet the index in the obs of the
        modality.
    
    component : str
        generative result of `modality` from which component to used.
    
    modality : str
        which modality to used from the generative result of 
        `component`.
    
    z_sampling_size: int, optional
        how many z to sample, by default 100
    
    x_sampling_size: int, optional
        how many x to sample, by default 1000

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    pd.DataFrame
        analysis result for differential analysis. The DataFrame 
        contains 4 columns, the first column specify the probability
        P(A>B|Z), the second column specify the probability P(B>A|Z),
        the third column specify the Bayesian factor of K(A>B|Z), the
        fourth column specify the Bayesian factor of K(B>A|Z).

    """
    x_means_a = []
    x_means_b = []
    x_sampling_size = min(len(group_a_index), len(group_b_index), x_sampling_size)
    for _ in tqdm(range(z_sampling_size)):
      mdata_group_a = self.sample_mdata_x(
          index=group_a_index,
          x_sampling_size=x_sampling_size)
      mdata_group_b = self.sample_mdata_x(
          index=group_b_index,
          x_sampling_size=x_sampling_size)
      dataloader_group_a = DataLoader(mdata_group_a, batch_size=batch_size)
      dataloader_group_b = DataLoader(mdata_group_b, batch_size=batch_size)
      batch_size = min(dataloader_group_a.batch_size, dataloader_group_b.batch_size)
      x_means_a.append(self.compute_x_means(
          dataset=dataloader_group_a.dataset,
          component=component,
          modality=modality,
          batch_size=batch_size))
      x_means_b.append(self.compute_x_means(
          dataset=dataloader_group_b.dataset,
          component=component,
          modality=modality,
          batch_size=batch_size))
    
    x_means_a = np.vstack(x_means_a)
    x_means_b = np.vstack(x_means_b)
    index = self.mdata.mod[modality].var.index 

    return self.compute_bayesian_factor(x_means_a, x_means_b, index)
  
  def sample_mdata_x(
      self,
      index: Union[pd.Index, Sequence[str]],
      x_sampling_size: int = 1000) -> mu.MuData:
    """sample x from the mdata of samples with provided index.

    Parameters
    ----------
    index : Union[pd.Index, Sequence[str]]
        the samples to be sampled from.

    x_sampling_size: int, optional
        how many x to sample, by default 1000

    Returns
    -------
    mu.MuData
        MuData with sampled data.

    """
    index_sampled = np.random.choice(index, x_sampling_size, replace=False)
    adata_dict = {mod: adata[index_sampled] for mod, adata in self.mdata.mod.items()}

    return mu.MuData(adata_dict)

  def compute_x_means(
      self,
      dataset: tf.data.Dataset,
      component: str,
      modality: str,
      batch_size: int = 128) -> np.ndarray:
    """Compute the means of generative data.

    Parameters
    ----------
    dataset : tf.data.Dataset
        input dataset.

    component : str
        generative result of `modality` from which component to used.
    
    modality : str
        which modality to used from the generative result of 
        `component`.

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        the means of the generative data distribution, where index i 
        specify the samples, index j specify the means of the data 
        distribution of the variables.

    """
    dist_x_z_name = self.model.components.get(component).distribution_names.get(modality)
    dist_x_z_class = ReflectionHandler.get_class_by_name(dist_x_z_name, 'distributions')
    x_means = []
    for batch in dataset.batch(batch_size):
      result = self.model(batch, training=True)
      x_parameters = result.get(
          f"{component}/{modality}/{Constants.MODEL_OUTPUTS_X_PARAMS}")
      dist_x_z = dist_x_z_class.from_parameterizer_output(x_parameters)
      x_means.append(dist_x_z.mean().numpy())
    return np.vstack(x_means)

  def compute_bayesian_factor(
      self,
      x_means_a: np.array,
      x_means_b: np.array,
      index: Union[pd.Index, Sequence[str]]) -> pd.DataFrame:
    """Compute the Bayesian factor of two groups.

    Parameters
    ----------
    x_means_a : np.array
        the means of the generative data distribution of the first 
        group.

    x_means_b : np.array
        the means of the generative data distribution of the second 
        group.

    index : Union[pd.Index, Sequence[str]]
        the index of the output DataFrame (var.index).

    Returns
    -------
    pd.DataFrame
        analysis result for differential analysis. The DataFrame 
        contains 4 columns, the first column specify the probability
        P(A>B|Z), the second column specify the probability P(B>A|Z),
        the third column specify the Bayesian factor of K(A>B|Z), the
        fourth column specify the Bayesian factor of K(B>A|Z).
        
    """
    p_a_gt_b = np.mean(x_means_a > x_means_b, 0)
    p_a_leq_b = 1.0 - p_a_gt_b
    bayesian_factor_a_gt_b = np.log(p_a_gt_b + 1e-7) - np.log(p_a_leq_b + 1e-7)
    p_b_gt_a = np.mean(x_means_b > x_means_a, 0)
    p_b_leq_a = 1.0 - p_b_gt_a
    bayesian_factor_b_gt_a = np.log(p_b_gt_a + 1e-7) - np.log(p_b_leq_a + 1e-7)

    return pd.DataFrame({
      'P(A>B|Z)': p_a_gt_b,
      'P(B>A|Z)': p_b_gt_a,
      'K(A>B|Z)': bayesian_factor_a_gt_b,
      'K(B>A|Z)': bayesian_factor_b_gt_a
    }, index=index)