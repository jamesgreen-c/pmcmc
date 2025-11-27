
from jax import Array, lax, vmap
import jax.random as jr
import jax.tree_util as jtu
import jax.numpy as jnp

from feynman_kac.utils import log_normalize, ess
from feynman_kac.protocol import FeynmacKac, PFConfig
from feynman_kac.base_particle_filter import BaseParticleFilter


class PF(BaseParticleFilter):
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


class CSMC(BaseParticleFilter):
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
    

class iRWCSMC(BaseParticleFilter):
    """
    Implements an iterated Random-Walk Conditional Sequential Monte Carlo (i-RW-CSMC).
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

    def t0(self, key: jr.PRNGKey, N, obs0: Array, ref: Array | tuple[Array]):

        assert ref is not None, "ConditionalBPF requires a reference trajectory."

        x_n0 = self.model.p0(key, N)  # (N,d)
        x_n0 = jtu.tree_map(lambda x, ref: x.at[0].set(ref[0]), x_n0, ref)  # set reference particle
        
        log_wn0 = self.vmap_log_p0(x_n0) + self.vmap_log_g(0, x_n0, x_n0, obs0)  # (N,)
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

        # add noise around reference particle
        D = x_n_prev.shape[-1]
        N = x_n_prev.shape[0]
        keys_d = jr.split(key1, D)
        Sig = 0.5 * (jnp.eye(N - 1) + jnp.ones((N - 1, N - 1)))  # (N-1, N-1)
        
        U = vmap(
            lambda k: jr.multivariate_normal(k, jnp.zeros((N - 1,)), Sig),
            in_axes=(0)
        )(keys_d).T  # (N-1, D)

        # moves are Gaussian noise around reference path at t
        ref_now = jtu.tree_map(lambda r: r[t], ref)     # extract PyTree slice
        U = vmap(
            lambda u: ref_now + 0.1 * (u / jnp.sqrt(D))
        )(U)  # (N-1, D)

        x_nt = jtu.tree_map(
            lambda x, u: x.at[1:].set(u),
            x_n_prev,
            U
        )
        x_nt = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_nt, ref_now)

        # calculate weights for current particles
        log_wnt = self.vmap_log_pt(t, x_nt, x_n_prev) + self.vmap_log_g(t, x_nt, x_n_prev, y_t)
        w_nt_norm, logZ_t = log_normalize(log_wnt)
        logZ = logZ_prev + logZ_t

        return (key2, x_nt, log_wnt, w_nt_norm, logZ, ref), (x_nt, w_nt_norm, idx, ess_t)
