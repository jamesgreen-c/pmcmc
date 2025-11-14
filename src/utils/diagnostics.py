import jax
import jax.numpy as jnp


def is_psd_cholesky(A: jnp.ndarray) -> bool:
    try:
        _ = jnp.linalg.cholesky(A + 1e-10 * jnp.eye(A.shape[0]))
        return True
    except jax.errors.ConcretizationTypeError:
        # when tracing under jit, don't use Python exceptions
        raise
    except Exception:
        return False
