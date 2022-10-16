from cavachon.dataloader.DataLoader import DataLoader
from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag
from cavachon.distributions.MixtureMultivariateNormalDiag import MixtureMultivariateNormalDiag
from tqdm import tqdm
from typing import Sequence

import muon as mu
import numpy as np
import pandas as pd
import scanpy
import tensorflow as tf

class ClusterAnalysis:
  def __init__(
      self,
      mdata: mu.MuData,
      model: tf.keras.Model):
    self.mdata = mdata
    self.model = model

  def compute_cluster_probability(
      self,
      modality: str,
      component: str,
      batch_size: int = 128) -> np.array:

    z_prior_parameterizer = self.model.components[component].z_prior_parameterizer
    z_prior_parameters = tf.squeeze(z_prior_parameterizer(tf.ones((1, 1))))
    dist_z_y = MultivariateNormalDiag.from_parameterizer_output(z_prior_parameters[..., 1:])  
    dist_z = MixtureMultivariateNormalDiag.from_parameterizer_output(z_prior_parameters)
    logpy = tf.math.log(tf.math.softmax(z_prior_parameters[..., 0]) + 1e-7)

    logpy_z = list()
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    for batch_data in tqdm(dataloader):
      outputs = self.model(batch_data, training=False)
      z = outputs[f'{component}/z']
      logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
      logpz = tf.expand_dims(dist_z.log_prob(z), -1)
      logpy_z.append(logpy + logpz_y - logpz)

    logpy_z = np.vstack(logpy_z)
    cluster = tf.argmax(logpy_z, axis=-1).numpy()
    self.mdata.mod[modality].obsm[f'logpy_z_{component}'] = logpy_z
    self.mdata.mod[modality].obs[f'cluster_{component}'] = [f'Cluster {x:03d}'for x in cluster]
    
    return logpy_z

  def compute_neighbors_with_same_annotations(
      self,
      modality: str,
      use_cluster: str,
      use_rep: str,
      n_neighbors_sequence: Sequence[int] = list(range(5, 25))) -> pd.DataFrame:
    if use_cluster not in self.mdata.mod[modality].obs:
      message = f'{use_cluster} not in obs DataFrame of the modality.'
      raise KeyError(message)
    
    proportions = list()
    clusters = list()
    n_neighbors = list()
    for k in tqdm(n_neighbors_sequence):
      scanpy.pp.neighbors(self.mdata[modality], n_neighbors=k + 1, use_rep=use_rep)
      proportions_k = list()
      for i, j in enumerate(self.mdata[modality].obsp['distances']):
        cluster = self.mdata[modality].obs.iloc[i][use_cluster]
        neighbor_clusters = self.mdata[modality].obs.iloc[j.indices][use_cluster]
        proportions_k.append((neighbor_clusters == cluster).sum()/ k)
        clusters += [cluster]
      
      proportions.append(np.array(proportions_k))
      n_neighbors.append(np.array([k] * len(proportions_k)))

    return pd.DataFrame({
        '% of KNN Cells with the Same Cluster': np.concatenate(proportions), 
        'Number of Neighbors': np.concatenate(n_neighbors),
        'Cluster': clusters})
      