import jax.numpy as jnp
import jax.random as jr
from jax import Array
import jax

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