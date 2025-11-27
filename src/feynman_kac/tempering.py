from jax import Array, lax, nn
import jax.random as jr
import jax.tree_util as jtu
import jax.numpy as jnp

from feynman_kac.utils import log_normalize, ess
from feynman_kac.protocol import FeynmacKac, PFConfig
from feynman_kac.base_particle_filter import BaseParticleFilter


class BaseTemperedPF(BaseParticleFilter):
    """
    Contains shared methods for all tempered particle filters
    """

    def __init__(self, model: FeynmacKac, cfg: PFConfig):
        super().__init__(model, cfg)

    def find_temper_delta(self, log_w_norm, log_g_t, beta, ess_target):
        left, right = 1e-3, 1.0 - beta

        def ess_at(delta):
            lw_new = log_w_norm + delta * log_g_t
            lw_new -= nn.logsumexp(lw_new)
            return ess(jnp.exp(lw_new)), lw_new

        def bisect(_):
            def cond(carry):
                l, r = carry
                return (r - l) > 1e-4

            def body(carry):
                l, r = carry
                mid = 0.5 * (l + r)
                ess_mid, _ = ess_at(mid)
                # ESS decreases with delta
                l = jnp.where(ess_mid > ess_target, mid, l)   # ESS too high → increase Δβ
                r = jnp.where(ess_mid <= ess_target, mid, r)  # ESS too low → shrink Δβ
                return (l, r)

            l, r = lax.while_loop(cond, body, (left, right))
            return 0.5 * (l + r)

        ess_left, _ = ess_at(left)
        ess_right, _ = ess_at(right)

        # jax.debug.print("[ESS] left={}, right={}", ess_left, ess_right)

        # handle edge cases based on monotonic ESS
        delta = lax.cond(
            ess_target >= ess_left,      # target ESS higher than achievable (Δβ=0)
            lambda _: left,
            lambda _: lax.cond(
                ess_target <= ess_right,  # target ESS too low (Δβ=max)
                lambda _: right,
                lambda _: bisect(_),
                operand=None
            ),
            operand=None
        )

        ess_val, lw_new = ess_at(delta)
        return delta, ess_val, lw_new
    
    def temper(self, key, x_nt, x_n_prev, t, y_t, ess_min, N):
        """
        
        1. Sample x_nt_b as x_nt_b | x_nb_prev ~ N(A @ x_nb_prev, Q / b). If b=0 use x_n_prev.
        2. Calculate tempered log potentials (beta @ log_g_curr) using x_nt_b and y_t. 
        3. If ESS < ESS_min: resample.
        4. Repeat until b = 1

        TODOs:
         - correct the doc string above
         - correct the lineage tracking through MCMC mutations and resampling.
        """

        beta = 0.0
        x_nt_b = x_nt

        # start with weights 1/N
        norm_wb = jnp.ones(N) / N
        log_wb = jnp.log(norm_wb) 
        log_g_curr = self.vmap_log_g(t, x_nt_b, x_n_prev, y_t)
        log_pt_curr = self.vmap_log_pt(t, x_nt_b, x_n_prev)
        
        def cond(carry):
            beta, *_ = carry
            return beta < 1.0 - 1e-9

        def step(carry):
            
            beta, key, x_nt_b, log_wb_prev, log_g_curr, log_pt_curr, _ = carry

            # 1. Find delta temperature
            delta, _, _ = self.find_temper_delta(log_wb_prev, log_g_curr, beta, ess_min)
            delta = jnp.minimum(delta, 1.0 - beta)
            beta = beta + delta

            # jax.debug.print("[temperature] beta={}", beta)

            # 2. Compute new weights
            # log_g_curr = self.vmapped_log_g(t, x_nt_b, x_n_prev, y_t)
            # log_pt_curr = self.vmapped_log_pt(t, x_nt_b, x_n_prev)
            log_wb = (delta * log_g_curr)   # + log_wb_prev  omitting this gives better results
            norm_wb, _ = log_normalize(log_wb)

            # 3. Potentially resample
            key, subkey = jr.split(key)
            x_nb_res, log_wb, idx, ess_tb = self.resampler(
                subkey, N, norm_wb, ess_min, x_nt_b, log_wb
            )
            log_g_curr = log_g_curr[idx]
            log_pt_curr = log_pt_curr[idx]

            # 4. Gaussian local proposal
            key, key_norm, key_unif = jr.split(key, 3)
            noise = 0.1 * jr.normal(key_norm, shape=x_nb_res.shape)
            x_nt_b_prop = x_nb_res + noise

            # 5. Calculate acceptances for proposals
            log_g_prop = self.vmap_log_g(t, x_nt_b_prop, x_n_prev, y_t)
            log_pt_prop = self.vmap_log_pt(t, x_nt_b_prop, x_n_prev)

            unif_n = jr.uniform(key_unif, shape=(x_nt_b.shape[0], ))
            logA = beta * (log_g_prop - log_g_curr) + (log_pt_prop - log_pt_curr)
            alpha = jnp.minimum(1.0, jnp.exp(logA))
            accept = unif_n <= alpha

            # update x and log potentials to reflect acceptances
            x_nb_next  = jnp.where(accept[:, None], x_nt_b_prop, x_nb_res)
            log_g_next  = jnp.where(accept, log_g_prop, log_g_curr)
            log_pt_next  = jnp.where(accept, log_pt_prop, log_pt_curr)
                
            # # reset weights
            # norm_wb = jnp.ones(N) / N
            # log_wb = jnp.log(norm_wb)
            
            return (beta, key, x_nb_next, log_wb, log_g_next, log_pt_next, idx)

        # run the temper scan
        beta, key, x_nt, log_wb, _, _, idx = lax.while_loop(
            cond,
            step,
            (beta, key, x_nt_b, log_wb, log_g_curr, log_pt_curr, jnp.arange(N))
        )
        return key, x_nt, log_wb, idx
    

class TemperedPF(BaseTemperedPF):
    """
    A Tempered Particle Filter (TPF) specialised to the Bootstrap proposal.

    This algorithm performs **incremental likelihood tempering** inside each PF
    time-step in order to prevent weight degeneracy and maintain a target
    effective sample size (ESS). At every observation y_t, a temperature
    schedule {β_k}_k is constructed adaptively via bisection so that:

        w_n(β + Δβ) ∝ w_n(β) * exp(Δβ * log g_t(x_n))

    where log g_t is the emission log-likelihood of the state x_n.

    After each tempering increment, particles are optionally resampled and
    locally rejuvenated using a Metropolis–Hastings mutation step. The final
    β=1 distribution coincides with the true filtering distribution at time t.
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

        # jax.debug.print("[step] t={}", t)

        # sample transition
        keys_t = jr.split(key1, self.N)
        x_nt = self.vmap_pt(keys_t, x_n_prev, t)  # (N,d)
        
        # tempering (does resampling and weight calculation)
        key_temper, x_nt, log_wnt, idx = self.temper(
            key2,           # RNG
            x_nt,           # current particles
            x_n_prev,       # previous particles
            t,              # time index
            y_t,            # observation
            self.ess_min,        # ESS threshold
            self.N,              # number of particles
        )
        
        w_nt_norm, logZ_t = log_normalize(log_wnt)
        logZ = logZ_prev + logZ_t
        
        # track ESS
        ess_t = ess(w_nt_norm)
        
        return (key_temper, x_nt, log_wnt, w_nt_norm, logZ, ref), (
            x_nt,
            w_nt_norm,
            idx,
            ess_t,
        )
    
    