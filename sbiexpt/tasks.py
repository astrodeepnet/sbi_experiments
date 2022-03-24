import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions



def equation(y,t, theta):
  """
  Evaluate the Lotka-Volterra time derivative of the solution `y` at time
  `t` as `func(y, t, *args)` to feed odeint.
  Parameters
  ----------
  y: float
    The two populations.
  t : float
    The time.
  theta: float
    Model parameters.
  Returns
  -------
    Lotka-Volterra equations.
  """

  X = y[0]
  Y = y[1]

  alpha, beta, gamma, delta = (
    theta[..., 0],
    theta[..., 1],
    theta[..., 2],
    theta[..., 3],
    )

  dX_dt = alpha * X - beta * X * Y
  dY_dt = -gamma * Y + delta * X * Y

  return jnp.stack([dX_dt, dY_dt],axis=0)



def LotkaVolterra(theta, init, key):
  """
  Generate observation x and compute log likelihood(x) + log prior(theta).
  Parameters
  ----------
  theta: float
    Model parameters.
  init:
    Initial conditions.
  key: PRNGKeyArray
  Returns
  -------
    Log propbability and observation.
  """

  ts = ts = jnp.arange(0.,20.,0.1)

  z = odeint(equation, init, ts, theta, rtol=1e-9, atol=1e-9)
  z = z.T[:,::21].reshape(1, -1)

  # prior over model parameters alpha, beta, gamma and delta
  prior = tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1)
  likelihood = tfd.Independent(tfd.LogNormal(jnp.log(z.T),0.1),2)

  proportion = likelihood.sample(seed=key)

  posterior = likelihood.log_prob(jax.lax.stop_gradient(proportion))
  posterior += prior.log_prob(theta)

  return posterior, proportion



@jax.jit
def get_batch_fixed_init_cond(key, batch_size=10000):
  """
  Generate dataset (theta, observation, score) with initial conditions [x0,y0] = [30,1].
  Parameters
  ----------
  key: PRNGKeyArray
  batch_size: int
    Size of the batch.
  Returns
  -------
    Dataset (theta, observation, score).
    'Theta': the model parameters.
    'Observation': the simulations from p(x|theta). The shape is (batch_size, time*2) with time = 10.
    'Score': the joint score Grad_theta(log p(theta|x,z)) with z the initial conditions.
  """

  key1, key2 = jax.random.split(key,2)

  theta = tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1).sample(batch_size, key1)
  init = jnp.array([30.,1.])*jnp.ones([batch_size,2])

  score, x = jax.vmap(jax.grad(lambda params, z, key : LotkaVolterra(params, z, key=key), has_aux=True))(theta, init,jax.random.split(key,batch_size))
  x = x.reshape(batch_size,-1)

  return theta, x, score


@jax.jit
def get_batch_NOTfixed_init_cond(key, batch_size=10000):
  """
  Generate dataset (theta, observation, score) with stochastic initial conditions sampled from LogNormal(log(10),0.8).
  Parameters
  ----------
  key: PRNGKeyArray
  batch_size: int
    Size of the batch.
  Returns
  -------
    Dataset (theta, observation, score).
    'Theta': the model parameters.
    'Observation': the simulations from p(x|theta). The shape is (batch_size, time*2) with time = 10.
    'Score': the joint score Grad_theta(log p(theta|x,z)) with z the initial conditions.
  """

  key1, key2 = jax.random.split(key,2)

  theta = tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1).sample(batch_size, key1)
  init = tfd.LogNormal(jnp.log(10*jnp.ones(2)), 0.8*jnp.ones(2)).sample(batch_size, key2)

  score, x = jax.vmap(jax.grad(lambda params, z, key : LotkaVolterra(params, z, key=key), has_aux=True))(theta, init,jax.random.split(key,batch_size))
  x = x.reshape(batch_size,-1)

  return theta, x, score
