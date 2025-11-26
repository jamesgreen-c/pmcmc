import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax
import jax

from feynman_kac.utils import _gather, ess

# TODO go over the resamplers in SMC book again and make sure they are correct


def single_multinomial(key: jr.PRNGKey, w: Array) -> Array:
    N = w.shape[0]
    return jr.choice(key, N, shape=(1,), replace=True, p=w)

def multinomial_resample(key: jr.PRNGKey, w: Array) -> Array:
    N = w.shape[0]
    return jr.choice(key, N, shape=(N,), replace=True, p=w)


def systematic_resample(key: jr.PRNGKey, w: Array) -> Array:
    N = w.shape[0]
    positions = (jr.uniform(key) + jnp.arange(N)) / N
    cdf = jnp.cumsum(w)
    return jnp.searchsorted(cdf, positions, side="right")


def stratified_resample(key: jr.PRNGKey, w: Array) -> Array:
    N = w.shape[0]
    u = jr.uniform(key, shape=(N,))
    positions = (u + jnp.arange(N)) / N
    cdf = jnp.cumsum(w)
    return jnp.searchsorted(cdf, positions, side="right")


def residual_resample(key: jr.PRNGKey, w: Array) -> Array:
    N = w.shape[0]
    Nw = N * w
    counts = jnp.floor(Nw).astype(int)
    R = N - counts.sum()
    idx = jnp.repeat(jnp.arange(N), counts)
    idx = jax.lax.cond(
        R > 0,
        lambda: jnp.concatenate([idx, jr.choice(key, N, shape=(R, ), replace=True, p=(Nw - counts) / R)]),
        lambda: idx
    )
    key = jr.split(key)[0]
    idx = jr.shuffle(key, idx)
    return idx


RESAMPLERS = {
    "multinomial": multinomial_resample,
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "residual": residual_resample,
}

class Resampler:

    def __init__(self, resampler_fn):
        self.resampler_fn = resampler_fn

    def __call__(self, key, N, norm_wn_prev, ess_min, x_n_prev, log_wn_prev):
        def do_resample(_):
            idx = self.resampler_fn(key, norm_wn_prev)
            x_res = _gather(x_n_prev, idx)
            return x_res, jnp.zeros(N), idx
        
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