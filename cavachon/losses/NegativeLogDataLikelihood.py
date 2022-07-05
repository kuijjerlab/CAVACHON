
from cavachon.losses.CustomLoss import CustomLoss
from cavachon.postprocess.PostprocessStep import PostprocessStep
from typing import Dict

import tensorflow as tf

class NegativeLogDataLikelihood(CustomLoss, tf.keras.losses.Loss):
  def __init__(self, module, modality_ordered_map, name='negative_log_data_likelihood', **kwargs):
    kwargs.setdefault('name', name)
    super().__init__(**kwargs)
    self.module = module
    self.modality_ordered_map = modality_ordered_map
    self.postprocess_steps: Dict[str, PostprocessStep] = dict()
    for modality_name, modality in modality_ordered_map.data.items():
      self.postprocess_steps.setdefault(modality_name, modality.postprocess_steps)
    self.cache: tf.Tensor = tf.zeros((1, ))

  def call(self, y_true, y_pred, sample_weight=None):
    x_parameters = y_pred.get('x_parameters')
    
    negative_log_data_likelihood = None

    for modality_name, modality in self.modality_ordered_map.data.items():
      # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] 
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
      # term (a): logpx_z
      for postprocess_step in self.postprocess_steps.get(modality_name, []):
        y_true = postprocess_step.execute(y_true)
      x = tf.sparse.to_dense(y_true.get(f'{modality_name}:matrix'))
      dist_x_z = modality.dist_cls(**x_parameters.get(modality_name))
      logpx_z = tf.reduce_sum(dist_x_z.log_prob(x), axis=-1)
      
      if negative_log_data_likelihood is None:
        negative_log_data_likelihood = -logpx_z
      else:
        negative_log_data_likelihood -= logpx_z

    self.cache = tf.reduce_mean(negative_log_data_likelihood)   

    return negative_log_data_likelihood

  def update_module(self, module):
    self.module = module
    return