import pytest

xfail = pytest.mark.xfail

import jax
import jax.numpy as jnp

import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

import numpy as np
from numpy.testing import assert_allclose

from sbiexpt.bijectors import ImplicitRampBijector


def _make_inv_bijector(rho):
  def fun(t, a, b, c):
    batch_size = len(t)
    bijector = ImplicitRampBijector(rho, 
                              jnp.ones((batch_size,1))*a,
                              jnp.ones((batch_size,1))*b,
                              jnp.ones((batch_size,1))*c)
    y = bijector.forward(t.reshape([batch_size, 1]))*1.     
    return bijector.inverse(y).squeeze()
  return fun

# Parameters of the bijectors to use for the tests
_test_params = [ {'a':2., 'b':0.25, 'c':0.1},
                 {'a':0.01, 'b':0.75, 'c':0.1},
                 {'a':1., 'b':0.05, 'c':0.99}]


def test_ramp_bijector_score():
  """ This test function just checks that we can compute the score 
  correctly on a Normalizing Flow using the ramp bijector.
  """
  batch_size = 100
  d = 2

  def log_prob_fn(x, a, b, c):
    flow = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=jnp.zeros(d)+0.5, scale_identity_multiplier=1.),
                                        bijector=ImplicitRampBijector(lambda x: x**3,
                                                                      jnp.ones((1,2))*a,
                                                                      jnp.ones((1,2))*b,
                                                                      jnp.ones((1,2))*c))
    return flow.log_prob(x)
  
  fake_data = jnp.zeros([batch_size, d])+0.5
  # With these parameters, the bijector should be the identity, so the log prob should be the same as that of the normal distribution
  # same for the score
  log_prob, score = jax.vmap(jax.value_and_grad(lambda x: log_prob_fn(x.reshape([1,2]), 0.001, 0.5, 1.).squeeze()))(fake_data)

  ref_log_prob, ref_score = jax.vmap(jax.value_and_grad(lambda x: tfd.MultivariateNormalDiag(loc=jnp.zeros(d)+0.5, scale_identity_multiplier=1.).log_prob(x)))(fake_data)

  assert_allclose(log_prob, ref_log_prob, rtol=1e-5, atol=1e-6)
  assert_allclose(score, ref_score, rtol=1e-5, atol=1e-6)


def test_ramp_bijector_inverse_cubic():
  """ Testing inverse of ramp bijector  for cubic ramp
  """
  batch_size = 100
  x = np.linspace(0.001,0.999,batch_size)
  
  test_inv = jax.jit(_make_inv_bijector(lambda x: x**3))
  for params in _test_params:
    assert_allclose(test_inv(x, **params), x, rtol=1e-5, atol=1e-6)


def test_ramp_bijector_inverse_quintic():
  """ Testing inverse of ramp bijector for quintic ramp
  """
  batch_size = 100
  x = np.linspace(0.001,0.999,batch_size)

  # Testing for quintic ramp
  test_inv = jax.jit(_make_inv_bijector(lambda x: x**5))
  for params in _test_params:
    assert_allclose(test_inv(x, **params), x, rtol=1e-5, atol=1e-6)

@xfail(reason="Ramp bijector still not working properly for non-odd order monomial ramps")
def test_ramp_bijector_inverse_exp():
  """ Testing inverse of ramp bijector for exponential ramp
  """
  batch_size = 100
  x = np.linspace(0.001,0.999,batch_size)

  # Testing for quintic ramp
  test_inv = jax.jit(_make_inv_bijector(lambda x: jnp.exp(-1./(2*x**2))))
  for params in _test_params:
    assert_allclose(test_inv(x, **params), x, rtol=1e-5, atol=1e-6)



