"""
Jax implentation of the bootstrap particle filter
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, lax, jit

from resample.resamplers import RESAMPLERS
from feynmac_kac.utils import log_normalize, ess
from feynmac_kac.protocol import FeynmacKac, PFConfig, PFOutputs, CSMC


class BootstrapParticleFilter:
    """
    The Bootstrap PF takes the potential function as the emission likelihood,
    and the Markov transition kernel as the proposal distribution. Unlike a guided PF,
    the Bootstrap PF cannot use lookahead information from the observations when proposing particles.
    """

    def __init__(self, model: FeynmacKac, config: PFConfig):
        self.model = model
        self.cfg = config
        self.key = config.key
        self.resampler = RESAMPLERS[config.resample_scheme]

        self.vmapped_pt = vmap(lambda key, x, t: self.model.pt(key, x, t), in_axes=(0, 0, None))
        self.vmapped_log_g = vmap(self.model.log_g, in_axes=(None, 0, 0, None))

    def filter(self, key: jr.PRNGKey, T: int, obs: Array) -> PFOutputs:
        """
        """

        N = self.cfg.N
        ess_min = self.cfg.ess_threshold * N if self.cfg.ess_threshold is not None else jnp.inf    

        # t = 0 
        key0, key = jr.split(key)
        x_n0, log_wn0, norm_w_n0, logZ_0 = self.t0(key0, N, obs[0])

        def step(carry, obs_t):
            
            key, x_n_prev, log_wn_prev, norm_wn_prev, logZ_prev = carry
            key1, key2 = jr.split(key)
            t, y_t = obs_t

            x_n_prev, log_wn_prev, idx, ess_t = self.resample(key2, N, norm_wn_prev, ess_min, x_n_prev, log_wn_prev)

            # sample transition
            keys_t = jr.split(key1, N)
            x_nt = self.vmapped_pt(keys_t, x_n_prev, t)  # (N,d)

            # w_t = w_{t-1} * g_t(x_t, x_{t-1})
            log_wnt = self.vmapped_log_g(t, x_nt, x_n_prev, y_t) + log_wn_prev
            w_nt_norm, logZ_t = log_normalize(log_wnt)
            logZ = logZ_prev + logZ_t

            return (key2, x_nt, log_wnt, w_nt_norm, logZ), (x_nt, w_nt_norm, idx, ess_t) # , logZ)
        
        obs = (jnp.arange(1, T), obs[1:])
        carry0 = (key, x_n0, log_wn0, norm_w_n0, logZ_0)
        (key, _, _, w_nT_norm, logZ_hat), (particles, weights, ancestors, ess_hist) = lax.scan(step, carry0, obs)

        # prepend t=0
        particles = jnp.concatenate([x_n0[None, ...], particles], axis=0)               # (T+1, N, d)
        weights = jnp.concatenate([norm_w_n0[None, ...], weights], axis=0)              # (T+1, N)
        ess_hist = jnp.concatenate([ess_hist, jnp.array([ess(w_nT_norm)])], axis=0)     # (T+1,)

        self.key = key

        return PFOutputs(
            particles=particles,
            weights=weights,
            ancestors=ancestors,
            logZ_hat=logZ_hat,
            ess_history=ess_hist,
        )
    
    def t0(self, key: jr.PRNGKey, N, obs0: Array):
        x_n0 = self.model.p0(key, N)  # (N,d)
        log_wn0 = self.vmapped_log_g(0, x_n0, jnp.zeros_like(x_n0), obs0)  # (N,)
        norm_w_n0, logZ_0 = log_normalize(log_wn0)
        return x_n0, log_wn0, norm_w_n0, logZ_0
    
    def resample(self, key, N, norm_wn_prev, ess_min, x_n_prev, log_wn_prev):
        def do_resample(_):
            idx = self.resampler(key, norm_wn_prev)
            return x_n_prev[idx], jnp.zeros(N), idx
        
        def skip_resample(_):
            return x_n_prev, log_wn_prev, jnp.arange(N)
        
        ess_t = ess(norm_wn_prev)
        x_n_prev, log_wn_prev, idx = lax.cond(
            ess_t < ess_min,
            do_resample,
            skip_resample,
            operand=None
        )
        return x_n_prev, log_wn_prev, idx, ess_t
    

class ConditionalBPF(BootstrapParticleFilter, CSMC):
    """
    Conditional Bootstrap Particle Filter
    Takes an immortal trajectory that must be included among the particles at each time step.
    We assume the immortal trajectory is the first particle.
    """
    
    def __init__(self, model: FeynmacKac, config: PFConfig):
        super().__init__(model, config)

    def csmc(self, key: jr.PRNGKey, obs: Array, x_imm: Array | None) -> PFOutputs:
        """
        """

        T = obs.shape[0]

        # if no immortal trajectory provided, run standard BPF
        if x_imm is None:
            key, subkey = jr.split(key)
            return self.filter(subkey, T, obs)
        
        N = self.cfg.N
        ess_min = self.cfg.ess_threshold * N if self.cfg.ess_threshold is not None else jnp.inf

        vmapped_pt = vmap(lambda key, x, t: self.model.pt(key, x, t), in_axes=(0, 0, None))
        vmapped_log_g = vmap(self.model.log_g, in_axes=(None, 0, 0, None))

        # t = 0 
        key0, key = jr.split(key)
        x_n0, log_wn0, norm_w_n0, logZ_0 = self.t0(key0, N, obs[0], x_imm)

        def step(carry, obs_t):
            
            key, x_n_prev, log_wn_prev, norm_wn_prev, logZ_prev = carry
            key1, key2 = jr.split(key)
            t, y_t = obs_t

            x_n_prev, log_wn_prev, idx, ess_t = self.resample(key2, N, norm_wn_prev, ess_min, x_n_prev, log_wn_prev)

            # restore immortal particle and correct ancestor index
            x_n_prev = x_n_prev.at[0].set(x_imm[t-1])
            idx = idx.at[0].set(0)

            # sample transition
            keys_t = jr.split(key1, N)
            x_nt = vmapped_pt(keys_t, x_n_prev, t)  # (N,d)
            x_nt = x_nt.at[0].set(x_imm[t])  # set immortal particle

            # w_t = w_{t-1} * g_t(x_t, x_{t-1})
            log_wnt = vmapped_log_g(t, x_nt, x_n_prev, y_t) + log_wn_prev
            w_nt_norm, logZ_t = log_normalize(log_wnt)
            logZ = logZ_prev + logZ_t

            return (key2, x_nt, log_wnt, w_nt_norm, logZ), (x_nt, w_nt_norm, idx, ess_t) # , logZ)
        
        obs = (jnp.arange(1, T), obs[1:])
        carry0 = (key, x_n0, log_wn0, norm_w_n0, logZ_0)
        (_, _, _, w_nT_norm, logZ_hat), (particles, weights, ancestors, ess_hist) = lax.scan(step, carry0, obs)

        # prepend t=0
        particles = jnp.concatenate([x_n0[None, ...], particles], axis=0)               # (T+1, N, d)
        weights = jnp.concatenate([norm_w_n0[None, ...], weights], axis=0)              # (T+1, N)
        ess_hist = jnp.concatenate([ess_hist, jnp.array([ess(w_nT_norm)])], axis=0)     # (T+1,)

        return PFOutputs(
            particles=particles,
            weights=weights,
            ancestors=ancestors,
            logZ_hat=logZ_hat,
            ess_history=ess_hist,
        )
    
    def t0(self, key: jr.PRNGKey, N, obs0: Array, x_imm: Array | None = None):

        if x_imm is None:
            x_n0 = self.model.p0(key, N)  # (N,d)
            log_wn0 = self.vmapped_log_g(0, x_n0, jnp.zeros_like(x_n0), obs0)  # (N,)
            norm_w_n0, logZ_0 = log_normalize(log_wn0)

        else:
            x_n0 = self.model.p0(key, N)  # (N,d)
            x_n0 = x_n0.at[0].set(x_imm[0])  # set immortal particle
            
            log_wn0 = self.vmapped_log_g(0, x_n0, jnp.zeros_like(x_n0), obs0)  # (N,)
            norm_w_n0, logZ_0 = log_normalize(log_wn0)
        
        return x_n0, log_wn0, norm_w_n0, logZ_0