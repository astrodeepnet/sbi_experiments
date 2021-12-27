import jax
import jax.numpy as jnp
from sbiexpt.implicit_inverse import make_inverse_fn
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

__all__ = ['ImplicitRampBijector']

class ImplicitRampBijector(tfp.bijectors.Bijector):
  """
  Bijector based on a ramp function, and implemented using an implicit 
  layer. 
  """

  def __init__(self, rho, a, b, c,  name = 'ImplicitRampBijector'):
    """
    Args:
      rho: function of x that defines a ramp function between 0 and 1
      a,b,c: scalar parameters of the coupling layer.
    """
    super(self.__class__, self).__init__(forward_min_event_ndims=0, name = name)
    self.a = a
    self.b = b
    self.c = c
    self.rho = rho
    self.sigma = lambda x : rho(x)/(rho(x)+rho(1-x))
    self.g = lambda x,a,b: self.sigma(a*(x-b)+0.5) 

    # Rescaled bijection
    def f(params, x):
      a, b, c = params
      b = (b - 1./(2*a))
      diff = x - b
      zs = jnp.stack([diff, -b*jnp.ones_like(x), 1 - b*jnp.ones_like(x)],axis=0)
      zs = zs * a
      y, y0, y1 = self.sigma(zs)
      y = (y - y0)/ (y1 - y0)
      return y*(1-c) + c *x
    self.f = f

    # Inverse bijector
    self.inv_f = make_inverse_fn(f)


  def _forward(self, x):
    return jax.vmap(self.f)([self.a, self.b, self.c], x)

  def _inverse(self, y):
      return jax.vmap(self.inv_f)([self.a, self.b, self.c], y)

  def _forward_log_det_jacobian(self, x):
    def logdet_fn(x,a,b,c):
      x = jnp.atleast_1d(x)
      g = jax.grad(self.f, argnums=1)([a,b,c], x)
      s, logdet = jnp.linalg.slogdet(jnp.atleast_2d([g]))
      return s*logdet
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c)


class AffineSigmoidBijector(tfp.bijectors.Bijector):
  """
  Bijector based on a ramp function, and implemented using an implicit 
  layer. 
  """

  def __init__(self, a, b, c,  name = 'AffineSigmoidBijector'):
    """
    Args:
      rho: function of x that defines a ramp function between 0 and 1
      a,b,c: scalar parameters of the coupling layer.
    """
    super(self.__class__, self).__init__(forward_min_event_ndims=0, name = name)
    self.a = a
    self.b = b
    self.c = c

    def sigmoid(x, a, b, c):
      z = jax.scipy.special.logit(x) * a + b
      y = jax.nn.sigmoid(z) * (1 - c) + c * x
      return y 

    # Rescaled bijection
    def f(params, x):
      a, b, c = params
      a_in, b_in = [0. - 1e-1, 1. + 1e-1]
      
      x = (x - a_in) / (b_in - a_in)
      x0 = (jnp.zeros_like(x) - a_in)/ ( b_in - a_in)
      x1 = (jnp.ones_like(x) - a_in) /( b_in - a_in)

      y = sigmoid(x, a, b, c)
      y0 = sigmoid(x0, a, b, c)
      y1 = sigmoid(x1, a, b, c)
      
      return (y - y0)/(y1 - y0)
    self.f = f

    # Inverse bijector
    self.inv_f = make_inverse_fn(f)

  def _forward(self, x):
    return jax.vmap(self.f)([self.a, self.b, self.c], x)

  def _inverse(self, y):
      return jax.vmap(self.inv_f)([self.a, self.b, self.c], y[...,0]).reshape(y.shape)

  def _forward_log_det_jacobian(self, x):
    def logdet_fn(x,a,b,c):
      g = jax.grad(self.f, argnums=1)([a,b,c], x[...,0])
      s, logdet = jnp.linalg.slogdet(jnp.atleast_2d(g))
      return s*logdet
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c)


class MixtureAffineSigmoidBijector(tfp.bijectors.Bijector):
  """
  Bijector based on a ramp function, and implemented using an implicit 
  layer. 

  This implementation is based on the Smooth Normalizing Flows described
  in: https://arxiv.org/abs/2110.00351

  """

  def __init__(self, a, b, c, p, name = 'MixtureAffineSigmoidBijector'):
    """
    Args:
      rho: function of x that defines a ramp function between 0 and 1
      a,b,c: scalar parameters of the coupling layer.
    """
    super(self.__class__, self).__init__(forward_min_event_ndims=0, name = name)
    self.a = a
    self.b = b
    self.c = c
    self.p = p

    def sigmoid(x, a, b, c):
      z = jax.scipy.special.logit(x) * a + b
      y = jax.nn.sigmoid(z) * (1 - c) + c * x
      return y 

    # Rescaled bijection
    def f(params, x):
      a, b, c, p = params
      a_in, b_in = [0. - 1e-1, 1. + 1e-1]
      x = x[...,jnp.newaxis]
      x = (x - a_in) / (b_in - a_in)
      x0 = (jnp.zeros_like(x) - a_in)/ ( b_in - a_in)
      x1 = (jnp.ones_like(x) - a_in) /( b_in - a_in)

      y = sigmoid(x, a, b, c)
      y0 = sigmoid(x0, a, b, c)
      y1 = sigmoid(x1, a, b, c)
      
      y = (y - y0)/(y1 - y0)

      return jnp.sum(p*(y*(1-c) + c *x), axis=0)
    self.f = f

    # Inverse bijector
    self.inv_f = make_inverse_fn(f)

  def _forward(self, x):
    return jax.vmap(self.f)([self.a, self.b, self.c, self.p], x[...,0]).reshape(x.shape)

  def _inverse(self, y):
      return jax.vmap(self.inv_f)([self.a, self.b, self.c, self.p], y[...,0]).reshape(y.shape)

  def _forward_log_det_jacobian(self, x):
    def logdet_fn(x,a,b,c,p):
      g = jax.grad(self.f, argnums=1)([a,b,c,p], x[...,0])
      s, logdet = jnp.linalg.slogdet(jnp.atleast_2d(g))
      return s*logdet
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c, self.p)
