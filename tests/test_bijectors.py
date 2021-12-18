import jax
import jax.numpy as jnp

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


def test_ramp_bijector_inverse_exp():
  """ Testing inverse of ramp bijector for exponential ramp
  """
  batch_size = 100
  x = np.linspace(0.001,0.999,batch_size)

  # Testing for quintic ramp
  test_inv = jax.jit(_make_inv_bijector(lambda x: jnp.exp(-1./(2*x**2))))
  for params in _test_params:
    assert_allclose(test_inv(x, **params), x, rtol=1e-5, atol=1e-6)



