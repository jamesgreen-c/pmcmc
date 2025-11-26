import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, lax
import jax.tree_util as jtu

from resample.resamplers import RESAMPLERS, Resampler
from feynman_kac.utils import ess
from feynman_kac.protocol import FeynmacKac, PFConfig, PFOutputs
from abc import ABC, abstractmethod


class BaseParticleFilter(ABC):
    """
    Abstract base class defining the generic structure of a particle filter.

    Responsibilities:
    - Store model, config, and RNG keys.
    - Expose common utilities: vmapped transition, vmapped likelihood, ESS, normalization.
    - Provide generic `t0()` and `resample()` methods, overridable by subclasses.
    - Provide a standard `scan_step()` signature that subclasses may override
      (e.g., ConditionalPF must inject an immortal path).
    - Provide a final `run_filter()` method that performs lax.scan:
        - calls t0()
        - loops using scan_step()
        - constructs PFOutputs
    """

    def __init__(self, model: FeynmacKac, cfg: PFConfig):

        # store model, config, and keys
        self.model = model
        self.cfg = cfg
        self.key = cfg.key

        # initialise resampler
        self.resampler = Resampler(RESAMPLERS[cfg.resample_scheme])

        # store common cfg vars
        self.N = cfg.N
        self.ess_min = self.cfg.ess_threshold * self.N if self.cfg.ess_threshold is not None else jnp.inf

        # expose common utilities
        self.vmap_pt = vmap(lambda k, x, t: self.model.pt(k, x, t), in_axes=(0, 0, None))
        self.vmap_log_g = vmap(self.model.log_g, in_axes=(None, 0, 0, None))

    def filter(
        self,
        key: jr.PRNGKey, 
        obs: Array, 
        ref: Array | tuple[Array] | None = None
    ) -> PFOutputs:
        """
        Generic particle filter implementation.

        params: Model parameters to use
        key: PRNGKey
        obs: (T, k) observations
        ref: Optional reference trajectory for conditional particle filtering
        """
        
        T = obs.shape[0]
        N = self.N
        
        # Initial t = 0 step 
        key0, key = jr.split(key)
        x_n0, log_wn0, norm_w_n0, logZ_0 = self.t0(key0, N, obs[0], ref)

        # create carry info
        obs = (jnp.arange(1, T), obs[1:])
        carry0 = (key, x_n0, log_wn0, norm_w_n0, logZ_0, ref)
        (_, _, _, w_nT_norm, logZ_hat, _), (particles, weights, ancestors, ess_hist) = lax.scan(self.step, carry0, obs)

        # prepend t=0
        particles = jtu.tree_map(
            lambda x0, xT: jnp.concatenate([x0[None, ...], xT], axis=0),
            x_n0, particles
        )                                                                               # (T, N, d)
        weights = jnp.concatenate([norm_w_n0[None, ...], weights], axis=0)              # (T, N)
        ess_hist = jnp.concatenate([ess_hist, jnp.array([ess(w_nT_norm)])], axis=0)     # (T,)

        return PFOutputs(
            particles=particles,
            weights=weights,
            ancestors=ancestors,
            logZ_hat=logZ_hat,
            ess_history=ess_hist,
        )

    def update_params(self, params: dict):
        """
        Update model parameters.
        """
        self.model.update(params)
    

    @abstractmethod
    def step(self, carry, obs_t):
        """
        One step of the particle filter.
        
        :param carry: The carried information from the previous step.
        :param obs_t: The observation at the current time step.
        """
        pass

    @abstractmethod
    def t0(self, key: jr.PRNGKey, N, obs0: Array, ref: Array | None = None):
        """
        Initialize the particle filter at time t=0.
        
        :param key: PRNGKey
        :param N: Number of particles
        :param obs0: Initial observation
        :param ref: Reference trajectory
        """
        pass
