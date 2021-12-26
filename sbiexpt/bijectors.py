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
      g = jax.grad(self.f, argnums=1)([a,b,c], x)
      s, logdet = jnp.linalg.slogdet(jnp.atleast_2d([g]))
      return s*logdet
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c)