from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from matplotlib import pyplot as plt

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother, lgssm_filter
from dynamax.utils.plotting import plot_uncertainty_ellipses


state_dim = 4
emission_dim = 2
delta = 1.0

# Create object
lgssm = LinearGaussianSSM(state_dim, emission_dim)

# Manually chosen parameters
initial_mean = jnp.array([8.0, 10.0, 1.0, 0.0])
initial_covariance = jnp.eye(state_dim) * 0.1
dynamics_weights  = jnp.array([[1, 0, delta, 0],
                               [0, 1, 0, delta],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
dynamics_covariance = jnp.eye(state_dim) * 0.001
emission_weights = jnp.array([[1.0, 0, 0, 0],
                              [0, 1.0, 0, 0]])
emission_covariance = jnp.eye(emission_dim)

# Initialize model
params, _ = lgssm.initialize(jr.PRNGKey(0),
                             initial_mean=initial_mean,
                             initial_covariance=initial_covariance,
                             dynamics_weights=dynamics_weights,
                             dynamics_covariance=dynamics_covariance,
                             emission_weights=emission_weights,
                             emission_covariance=emission_covariance)

num_timesteps = 15
key = jr.PRNGKey(300)
x, y = lgssm.sample(params, key, num_timesteps)

# Plot Data
observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
fig1, ax1 = plt.subplots()
ax1.plot(*x[:, :2].T, marker="s", color="C0", label="true state")
ax1.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions")
ax1.legend(loc="upper left")

lgssm_posterior = lgssm_filter(params, y)
print(lgssm_posterior.filtered_means.shape)
print(lgssm_posterior.filtered_covariances.shape)
print(lgssm_posterior.marginal_loglik)

fig2, ax2 = plt.subplots()
ax2.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
ax2.plot(*x[:, :2].T, ls="--", color="darkgrey", label="true state")
ax2.plot(*lgssm_posterior.filtered_means[:, :2].T, color="C1", label="filtered mean")
plot_uncertainty_ellipses(
    lgssm_posterior.filtered_means,
    lgssm_posterior.filtered_covariances,
    ax2,
    label="filtered means",
)