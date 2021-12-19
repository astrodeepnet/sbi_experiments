import jax
from jax import lax
import jax.numpy as jnp
from jaxopt import Bisection
from functools import partial
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

__all__ = ['ImplicitRampBijector']

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fixed_point_layer(f, params):
  z_star = tfp.math.find_root_chandrupatla(lambda z: f(params, z), low=0., high=1.).estimated_root
  print('tu', z_star.shape)
  return z_star

def fixed_point_layer_fwd(f, params):
  z_star = fixed_point_layer(f, params)
  return z_star, (params, z_star)

def fixed_point_layer_bwd(f, res, z_star_bar):
  x, z_star = res
  _, vjp_a = jax.vjp(lambda x: f(x, z_star), x)
  _, vjp_z = jax.vjp(lambda z: f(x, z), z_star)
  print('to',z_star_bar.shape,x[0].shape, z_star.shape)
  y = tfp.math.find_root_chandrupatla(lambda u: vjp_z(jnp.atleast_1d(u))[0] + z_star_bar).estimated_root
  print('tdu', y.shape)
  return vjp_a(y)

fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

class ImplicitRampBijector(tfp.bijectors.Bijector):
  """
  Bijector based on a ramp function, and implemented using an implicit 
  layer. 

  This implementation is based on the Smooth Normalizing Flows described
  in: https://arxiv.org/abs/2110.00351

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
    def f(x, a, b, c):
      b = (b - 1./(2*a))
      diff = x - b
      zs = jnp.stack([diff, -b*jnp.ones_like(x), 1 - b*jnp.ones_like(x)],axis=0)
      zs = zs * a
      y, y0, y1 = self.sigma(zs)
      y = (y - y0)/ (y1 - y0)
      return y*(1-c) + c *x
    self.f = f

    # Defining inverse bijection
    def fun(params, x):
      a,b,c,y = params
      return self.f(x,a,b,c) - y
    # Inverse bijector
    self.inv_f = lambda x,a,b,c: fixed_point_layer(fun, (a,b,c,x))

  def _forward(self, x):
    return self.f(x, self.a, self.b, self.c)

  def _inverse(self, y):
      return jax.vmap(self.inv_f)(y, self.a, self.b, self.c)

  def _forward_log_det_jacobian(self, x):
    x = x.reshape(self.b.shape)
    def logdet_fn(x,a,b,c):
      x = jnp.atleast_1d(x)
      jac = jax.jacobian(self.f, argnums=0)(x,a,b,c)
      s, logdet = jnp.linalg.slogdet(jac)
      return jnp.atleast_1d(s*logdet)
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c)


class MixtureImplicitRampBijector(tfp.bijectors.Bijector):
  """
  Bijector based on a ramp function, and implemented using an implicit 
  layer. 

  This implementation is based on the Smooth Normalizing Flows described
  in: https://arxiv.org/abs/2110.00351

  """

  def __init__(self, rho, a, b, c, p, name = 'MixtureImplicitRampBijector'):
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
    self.rho = rho
    self.sigma = lambda x : rho(x)/(rho(x)+rho(1-x))
    self.g = lambda x,a,b: self.sigma(a*(x-b)+0.5) 

    # Rescaled bijection
    def f(x, a, b, c, p):
      x = jnp.atleast_1d(x)
      x = x[jnp.newaxis,:]
      b = (b - 1./(2*a))
      diff = x - b
      zs = jnp.stack([diff, -b*jnp.ones_like(x), 1 - b*jnp.ones_like(x)],axis=0)
      zs = zs * a
      y, y0, y1 = self.sigma(zs)
      y = (y - y0)/ (y1 - y0)
      return jnp.sum(p[..., jnp.newaxis]*(y*(1-c) + c *x), axis=0)
    self.f = f

    # Defining inverse bijection
    def fun(x, aux):
      a,b,c,p,y = aux
      return jnp.squeeze(self.f(x,a,b,c,p) - y)
    def inv_f(x,a,b,c,p):
      bisec = Bisection(optimality_fun=fun, lower=0.01, upper=0.99, check_bracket=False, unroll=True)
      return bisec.run(aux=(a,b,c, p,x)).params
    # Inverse bijector
    self.inv_f = inv_f

  def _forward(self, x):
    return jax.vmap(self.f)(x, self.a, self.b, self.c, self.p)

  def _inverse(self, y):
      return jax.vmap(self.inv_f)(y, self.a, self.b, self.c, self.p).reshape(y.shape)

  def _forward_log_det_jacobian(self, x):
    x = x.reshape([self.b.shape[0], self.b.shape[-1]])
    def logdet_fn(x,a,b,c,p):
      x = jnp.atleast_1d(x)
      jac = jax.jacobian(self.f, argnums=0)(x,a,b,c,p)
      s, logdet = jnp.linalg.slogdet(jac)
      return jnp.atleast_1d(s*logdet)
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c, self.p)