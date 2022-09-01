from cavachon.modules.parameterizers.Parameterizer import Parameterizer

class MultivariateNormalDiag(Parameterizer):

  default_libsize_scaling = False

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'multivariate_normal_diag',
      libsize_scaling: bool = False,
      exp_transform: bool = False):

    return super().make(
      input_dims=input_dims,
      event_dims=event_dims,
      name=name,
      libsize_scaling=libsize_scaling,
      exp_transform=exp_transform)