#%%
import tensorflow_probability as tfp

class IndependentBernoulli(tfp.layers.DistributionLambda):
  def __init__(self, **kwargs):
    self.make_distribution_fn=lambda x: tfp.distributions.Bernoulli(logits=x.get('pi_logit'))
    convert_to_tensor_fn=lambda x: x.sample()
    super().__init__(
        make_distribution_fn=self.make_distribution_fn,
        convert_to_tensor_fn=convert_to_tensor_fn,
        **kwargs)

  def call(self, inputs, training=True):
    if not training:
      return self.make_distribution_fn(inputs).sample(), _
    else:
      return self.make_distribution_fn(inputs).sample(), _

# %%
import tensorflow as tf
x = IndependentBernoulli()
#x(tf.random.normal((1, 5)))
x({'pi_logit': tf.random.normal((1, 5))})

# %%
class TestLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()

  def call(self, y_true, y_pred):
    return tf.reduce_mean(y_pred, axis=1)

tmploss = TestLoss()
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, activation='relu')
])
# %%
model.compile(
  loss=tmploss
)
# %%
X = tf.random.normal((50, 20))
y = tf.random.normal((50, 1))
#ds = tf.data.Dataset.from_tensor_slices([X, y])
model.fit(x=X, y=y)
# %%
