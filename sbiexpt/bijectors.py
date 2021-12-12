import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

__all__ = ['ImplicitRampBijector']


@partial(jax.jit, static_argnums=(0,))
def fwd_solver(f, z_init):
  def cond_fun(carry):
    z_prev, z = carry
    return jnp.linalg.norm(z_prev - z) > 1e-4

  def body_fun(carry):
    _, z = carry
    return z, f(z)

  init_carry = (z_init, f(z_init))
  _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
  return z_star

def newton_solver(f, z_init):
  f_root = lambda z: f(z)
  g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
  return fwd_solver(g, z_init)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f, params, b):
  z_star = solver(lambda z: f(params, z), z_init=jnp.ones(1)*b) # initialized within [-1/2/a + b, 1/2/a + b]
  return z_star[0] # output needs to be a scalar in order to compute gradient in JAX

def fixed_point_layer_fwd(solver, f, params, b):
  z_star = fixed_point_layer(solver, f, params, b)
  return z_star, (params, z_star)

def fixed_point_layer_bwd(solver, f, res, z_star_bar):
  x, z_star = res
  z_star = z_star.reshape(-1) # needs to be an array to compute jacobian later
  _, vjp_a = jax.vjp(lambda x: f(x, z_star), x)
  _, vjp_z = jax.vjp(lambda z: f(x, z), z_star)
  return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
                      z_init=jnp.ones_like(z_star)
                      ))

fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

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
    super(ImplicitRampBijector, self).__init__(forward_min_event_ndims=0, name = name)
    self.a = a
    self.b = b
    self.c = c
    self.rho = rho
    self.sigma = lambda x : rho(x)/(rho(x)+rho(1-x))
    self.g = lambda x,a,b: self.sigma(a*(x-b)+0.5) 
    # Rescaled bijection
    self.f = lambda x,a,b,c: (1-c)*((self.g(x,a,b)-self.g(0.,a,b))/(self.g(1.,a,b)-self.g(0.,a,b)))+c*x
    # Defining inverse bijection
    def fun(params, x):
      a,b,c,y = params
      return self.f(x,a,b,c) - y
    self.inv_f = lambda x,a,b,c: fixed_point_layer(newton_solver, fun, (a,b,c,x), b)

  def _forward(self, x):
    return self.f(x, self.a, self.b, self.c)

  def _inverse(self, y):
    return jax.vmap(self.inv_f)(y, self.a, self.b, self.c).reshape((-1,1)) # TODO: fix for higher dimensions
  
  def _forward_log_det_jacobian(self, x):
    def logdet_fn(x,a,b,c):
      x = jnp.atleast_1d(x)
      jac = jax.jacobian(self.f, argnums=0)(x,a,b,c)
      s, logdet = jnp.linalg.slogdet(jac)
      return s*logdet
    return jax.vmap(logdet_fn)(x, self.a, self.b, self.c)