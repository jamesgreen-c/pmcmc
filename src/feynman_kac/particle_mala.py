
from jax import Array, lax, vmap, value_and_grad
import jax.random as jr
import jax.tree_util as jtu
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from feynman_kac.utils import log_normalize, ess
from feynman_kac.protocol import FeynmacKac, PFConfig
from feynman_kac.base_particle_filter import BaseParticleFilter



class ParticleAuxiliaryMALA(BaseParticleFilter):
    """
    Implements Particle-aMALA from Axel Finke 2024
    """

    def __init__(self, model, cfg, delta: float = 0.1):
        super().__init__(model, cfg)

        self.grad_log_Qt = vmap(
            value_and_grad(
                lambda t, x, xp, y: jnp.sum(self.model.log_g(t, x, xp, y) + self.model.log_pt(t, x, xp)),  # sum gets to scalar
                argnums=1 # gradient w.r.t. x
                ),
            in_axes=(None, 0, 0, None)
        )
        self.delta = delta


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
        ref_now = jtu.tree_map(lambda r: r[t], ref)     # extract PyTree slice        
        x_n_prev = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_n_prev, ref_prev)
        idx = idx.at[0].set(0)

        # add noise around reference particle with gradient info 
        delta = self.delta / 2
        keys = jr.split(key1, self.N + 1)
        _, grad_log_Qt_ref = self.grad_log_Qt(t, ref_now[None, ...], ref_prev[None, ...], y_t)
        grad_log_Qt_ref = jtu.tree_map(lambda g: g[0], grad_log_Qt_ref)  # squeeze out batch dim

        # sample aux vector
        Ut = jr.multivariate_normal(
            keys[0],
            ref_now + delta * grad_log_Qt_ref, 
            delta * jnp.eye(x_n_prev.shape[-1])
        )  # (D,)
        x_nt = vmap(
            lambda k: jr.multivariate_normal(k, Ut, delta * jnp.eye(x_n_prev.shape[-1])),
        )(keys[1:])  # (N, d)

        # keep reference particle fixed
        x_nt = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_nt, ref_now)

        # calculate MALA weights for current particles
        log_Qnt, grad_log_Qnt = self.grad_log_Qt(t, x_nt, x_n_prev, y_t)
        log_wnt = vmap(
            lambda log_Q, grad_log_Q, x: (
                log_Q 
                 + multivariate_normal.logpdf(Ut, x + delta * grad_log_Q, delta * jnp.eye(x_n_prev.shape[-1])) 
                  - multivariate_normal.logpdf(Ut, x, delta * jnp.eye(x_n_prev.shape[-1]))
            ),
            in_axes=(0, 0, 0)
        )(log_Qnt, grad_log_Qnt, x_nt)

        # normalise weights and calc logZ
        # log_wnt = log_wnt # + log_wn_prev
        w_nt_norm, logZ_t = log_normalize(log_wnt)
        logZ = logZ_prev + logZ_t

        return (key2, x_nt, log_wnt, w_nt_norm, logZ, ref), (x_nt, w_nt_norm, idx, ess_t)



class ParticleMALA(BaseParticleFilter):
    """
    Implements Particle-MALA from Axel Finke 2024
    """

    def __init__(self, model, cfg, delta: float = 0.1):
        super().__init__(model, cfg)

        self.delta = delta

        self.grad_log_Qt = vmap(
            value_and_grad(
                lambda t, x, xp, y: jnp.sum(self.model.log_g(t, x, xp, y) + self.model.log_pt(t, x, xp)),  # sum gets to scalar
                argnums=1 # gradient w.r.t. x
                ),
            in_axes=(None, 0, 0, None)
        )
        self.log_H = vmap(
            lambda grad_log_Qn, xn, xbar: (delta * grad_log_Qn.T @ (xbar - xn) - self.N * grad_log_Qn.T @ grad_log_Qn / (self.N + 1) ) / delta,
            in_axes=(0, 0, None) 
        )

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
        ref_now = jtu.tree_map(lambda r: r[t], ref)     # extract PyTree slice        
        x_n_prev = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_n_prev, ref_prev)
        idx = idx.at[0].set(0)

        # add noise around reference particle with gradient info 
        delta = self.delta / 2
        keys = jr.split(key1, self.N + 1)
        _, grad_log_Qt_ref = self.grad_log_Qt(t, ref_now[None, ...], ref_prev[None, ...], y_t)
        grad_log_Qt_ref = jtu.tree_map(lambda g: g[0], grad_log_Qt_ref)  # squeeze out batch dim

        # sample aux vector
        Ut = jr.multivariate_normal(
            keys[0],
            ref_now + delta * grad_log_Qt_ref, 
            delta * jnp.eye(x_n_prev.shape[-1])
        )  # (D,)
        x_nt = vmap(
            lambda k: jr.multivariate_normal(k, Ut, delta * jnp.eye(x_n_prev.shape[-1])),
        )(keys[1:])  # (N, d)

        # keep reference particle fixed
        x_nt = jtu.tree_map(lambda x, rv: x.at[0].set(rv), x_nt, ref_now)


        # calculate MALA weights for current particles
        xt_bar = x_nt.mean()
        log_Qnt, grad_log_Qnt = self.grad_log_Qt(t, x_nt, x_n_prev, y_t)
        log_wnt = log_Qnt + self.log_H(grad_log_Qnt, x_nt, xt_bar)

        # normalise weights and calc logZ
        # log_wnt = log_wnt # + log_wn_prev
        w_nt_norm, logZ_t = log_normalize(log_wnt)
        logZ = logZ_prev + logZ_t

        return (key2, x_nt, log_wnt, w_nt_norm, logZ, ref), (x_nt, w_nt_norm, idx, ess_t)
