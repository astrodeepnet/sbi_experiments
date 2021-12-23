import jax.numpy as np
from sklearn.datasets import make_swiss_roll
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

__all__ = ['get_swiss_roll', 'get_two_moons']


def get_swiss_roll(sigma, resolution=1024):
  """
  Generate a TFP approximate distribution of the swiss roll dataset

  Parameters
  ----------
  sigma: float
    Spread of the 2 moons distribution.
  resolution: int
    Number of components in the gaussian mixture approximation of the
    distribution (default: 1024)

  Returns
  -------
  distribution: TFP distribution
    Swiss roll distribution
  """
  n_samples = 2 * resolution
  X, _ = make_swiss_roll(n_samples, noise=0)
  coords = np.vstack([X[:, 0], X[:, 2]])

  distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution) / resolution / 2),
    components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
  )
  return distribution

def get_two_moons(sigma, resolution=1024, normalized=False):

  """
  Generate a TFP approximate distribution of the two moons dataset
  Parameters
  ----------
  sigma: float
    Spread of the 2 moons distribution.
  resolution: int
    Number of components in the gaussian mixture approximation of the
    distribution (default: 1024)
  normalized: bool
    Whether to recenter the distribution on [0,1]
  Returns
  -------
  distribution: TFP distribution
    Two moons distribution
  """

  outer_circ_x = np.cos(np.linspace(0, np.pi, resolution))
  outer_circ_y = np.sin(np.linspace(0, np.pi, resolution))
  inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, resolution))
  inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, resolution)) - .5

  X = np.append(outer_circ_x, inner_circ_x)
  Y = np.append(outer_circ_y, inner_circ_y)

  coords = np.vstack([X,Y])
  if normalized:
    coords = coords / 5 + 0.45

  distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution) / resolution / 2),
    components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
  )
  return distribution