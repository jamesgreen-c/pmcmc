
import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, lax, jit
import jax
import jax.tree_util as jtu

from resample.resamplers import RESAMPLERS, Resampler
from feynmac_kac.utils import log_normalize, ess
from feynmac_kac.protocol import FeynmacKac, PFConfig, PFOutputs
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


class BPF(BaseParticleFilter):
    """
    Implements the standard bootstrap particle filter.

    Overrides:
    - step(): normal bootstrap step (propagate + weight).
    - t0(): initial propagation and weighting for t=0.

    Responsibilities:
    - No conditional trajectories.
    - Resampling triggered only by ESS threshold.
    - Defines the standard BPF recursion.
    """
    
    
    def __init__(self, model: FeynmacKac, cfg: PFConfig):
        super().__init__(model, cfg)

    def t0(self, key: jr.PRNGKey, N, obs0: Array, ref: Array | tuple[Array] | None = None):
        x_n0 = self.model.p0(key, N)  # (N,d)
        log_wn0 = self.vmap_log_g(0, x_n0, None, obs0)  # (N,)
        norm_w_n0, logZ_0 = log_normalize(log_wn0)
        return x_n0, log_wn0, norm_w_n0, logZ_0

    def step(self, carry, obs_t):

        key, x_n_prev, log_wn_prev, norm_wn_prev, logZ_prev, ref = carry
        
        key1, key2 = jr.split(key)
        t, y_t = obs_t

        # jax.debug.print("[Step] t:{}", t)
        # resample
        x_n_prev, log_wn_prev, idx, ess_t = self.resampler(
            key2, 
            self.N, 
            norm_wn_prev, 
            self.ess_min, 
            x_n_prev, 
            log_wn_prev
        )

        # sample transition
        keys_t = jr.split(key1, self.N)
        x_nt = self.vmap_pt(keys_t, x_n_prev, t)  # (N,d)

        # w_t = w_{t-1} * g_t(x_t, x_{t-1})
        log_wnt = self.vmap_log_g(t, x_nt, x_n_prev, y_t) + log_wn_prev
        w_nt_norm, logZ_t = log_normalize(log_wnt)
        logZ = logZ_prev + logZ_t

        return (key2, x_nt, log_wnt, w_nt_norm, logZ, ref), (x_nt, w_nt_norm, idx, ess_t)


class ConditionalBPF(BaseParticleFilter):
    """
    Implements a Conditional Sequential Monte Carlo (CSMC) / Conditional BPF.

    Responsibilities:
    - Keep an 'reference' trajectory fixed through time.
    - Override:
        - t0(): inject ref[0] into the particle set.
        - step(): ensure particle 0 always follows the fixed path.
                  ensure ancestor index 0→0 after resampling.
    - All other logic identical to BaseParticleFilter.
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
    
    def t0(self, key: jr.PRNGKey, N, obs0: Array, ref: Array | tuple[Array]):

        assert ref is not None, "ConditionalBPF requires a reference trajectory."

        x_n0 = self.model.p0(key, N)  # (N,d)
        x_n0 = jtu.tree_map(lambda x, ref: x.at[0].set(ref[0]), x_n0, ref)  # set reference particle
        
        log_wn0 = self.vmap_log_g(0, x_n0, x_n0, obs0)  # (N,)
        norm_w_n0, logZ_0 = log_normalize(log_wn0)
    
        return x_n0, log_wn0, norm_w_n0, logZ_0

    def step(self, carry, obs_t):
            
            key, x_n_prev, log_wn_prev, norm_wn_prev, logZ_prev, ref = carry
            key1, key2 = jr.split(key)
            t, y_t = obs_t

            x_n_prev, log_wn_prev, idx, ess_t = self.resampler(key2, self.N, norm_wn_prev, self.ess_min, x_n_prev, log_wn_prev)

            # restore immortal particle and correct ancestor index
            ref_prev = jtu.tree_map(lambda r: r[t-1], ref)     # extract PyTree slice
            x_n_prev = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_n_prev, ref_prev)
            idx = idx.at[0].set(0)

            # sample transition
            keys_t = jr.split(key1, self.N)
            x_nt = self.vmap_pt(keys_t, x_n_prev, t)  # (N,d)
            ref_now = jtu.tree_map(lambda r: r[t], ref)
            x_nt = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_nt, ref_now)

            # w_t = w_{t-1} * g_t(x_t, x_{t-1})
            log_wnt = self.vmap_log_g(t, x_nt, x_n_prev, y_t) + log_wn_prev
            w_nt_norm, logZ_t = log_normalize(log_wnt)
            logZ = logZ_prev + logZ_t

            return (key2, x_nt, log_wnt, w_nt_norm, logZ, ref), (x_nt, w_nt_norm, idx, ess_t)