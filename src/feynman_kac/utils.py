import jax.numpy as jnp
import jax.tree_util as jtu


def log_normalize(logw: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Return normalized weights and log-mean-exp (for logZ increment)."""
    m = jnp.max(logw)
    w = jnp.exp(logw - m)
    Z = w.sum()
    logZ_inc = jnp.log(Z) + m
    return (w / (w.sum() + 1e-30)), logZ_inc


def ess(w: jnp.ndarray) -> float:
    """
    w: (n, )
    """
    return 1.0 / jnp.sum(jnp.square(w) + 1e-30)


def _gather(x, idx):
    return jtu.tree_map(lambda a: a[idx], x)

