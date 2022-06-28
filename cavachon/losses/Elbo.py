from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper

import tensorflow as tf
import tensorflow_probability as tfp

class Elbo(tf.keras.losses.Loss):
  def __init__(self, module, modality_ordered_map, **kwargs):
    super().__init__(**kwargs)
    #self.alpha = kwargs.get('alpha', 1.0)
    #self.beta = kwargs.get('beta', 1.0)
    self.module = module
    self.modality_ordered_map = modality_ordered_map

  def call(self, y_true, y_pred, sample_weight=None):
    z_parameters = y_pred.get('z_parameters')
    z = y_pred.get('z')
    x_parameters = y_pred.get('x_parameters')
    
    elbo = None
    data_likelihood = None
    kl_divergence = None

    for modality_name, modality in self.modality_ordered_map.data.items():
      # We use y to denote c_j
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] eq (C.48) from Falck et al., 2021.
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
      # term (a): logpx_z
      x = tf.sparse.to_dense(y_true.get(f'{modality_name}:matrix'))
      dist_x_z = modality.dist_cls(**x_parameters.get(modality_name))
      logpx_z = tf.reduce_sum(dist_x_z.prob(x), axis=-1)

      elbo = logpx_z if elbo is None else elbo + logpx_z
      data_likelihood = logpx_z if data_likelihood is None else data_likelihood + logpx_z

      z_prior = self.module.z_prior.get(modality_name)
      dist_z_x = MultivariateNormalDiagWrapper(**z_parameters.get(modality_name))
      dist_z_y = MultivariateNormalDiagWrapper(
          tf.transpose(z_prior.mean_z_y, [1, 0]),
          tf.transpose(z_prior.var_z_y, [1, 0])
      )
      dist_y = tfp.distributions.Categorical(logits=tf.squeeze(z_prior.pi_logit_y), allow_nan_stats=False)
      dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y.dist)
      
      logpz_y = dist_z_y.log_prob(tf.expand_dims(z.get(modality_name), -2))
      logpy = tf.math.log(z_prior.pi_y + 1e-7)
      logpz = dist_z.log_prob(tf.expand_dims(z.get(modality_name), -2))
      
      logpy_z = logpz_y + logpy - logpz
      py_z = tf.exp(logpy_z)
      
      # term (b): 𝚺_j𝚺_y[py_z(logpz_y)]
      py_z_logpz_y = tf.reduce_sum(py_z * logpz_y, axis=-1)

      # term (c): 𝚺_j𝚺_y[py_z(logpy)]
      py_z_logpy = tf.reduce_sum(py_z * logpy, axis=-1)

      # term (d): 𝚺_j[logqz_x]
      logqz_x = dist_z_x.log_prob(z.get(modality_name))
      
      # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
      py_z_logpy_z = tf.reduce_sum(py_z * logpy_z, axis=-1)
  
      kl_divergence = py_z_logpz_y if kl_divergence is None else py_z_logpz_y
      kl_divergence += py_z_logpz_y
      kl_divergence += py_z_logpy
      kl_divergence += -py_z_logpy_z
      kl_divergence += -logqz_x

      elbo += kl_divergence

    return elbo
    
  def results(self):
    return
  