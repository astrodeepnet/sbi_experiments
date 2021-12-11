import jax
import jax.numpy as np
from sklearn.datasets import make_swiss_roll
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

def get_swiss_roll(sigma, resolution=1024):
  """
  Generate a swiss roll dataset.
  """
  n_samples = 2*resolution
  X, _ = make_swiss_roll(n_samples, noise=0)
  coords = np.vstack([X[:, 0], X[:, 2]])

  distribution = tfd.MixtureSameFamily(
  mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution)/resolution/2),
  components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
  )
  return distribution

def get_two_moons(sigma, resolution=1024):
  """
  Returns two moons distribution as a TFP distribution
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
  Two moon distribution
  """

  outer_circ_x = np.cos(np.linspace(0, np.pi, resolution))
  outer_circ_y = np.sin(np.linspace(0, np.pi, resolution))
  inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, resolution))
  inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, resolution)) - .5

  X = np.append(outer_circ_x, inner_circ_x)
  Y = np.append(outer_circ_y, inner_circ_y)
  coords = np.vstack([X,Y])

  distribution = tfd.MixtureSameFamily(
  mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution)/resolution/2),
  components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
  )
  return distribution