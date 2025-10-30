import jax.numpy as jnp
from jax.scipy.special import gammaln

def log_multivariate_gamma(p, a):
    """Log multivariate gamma function log Γ_p(a)."""
    i = jnp.arange(p)
    return (p*(p-1)/4)*jnp.log(jnp.pi) + jnp.sum(gammaln(a - i/2))

def invwishart_logpdf(Q, Psi, nu):
    """
    Log pdf of Inverse-Wishart(Q | Psi, nu)
    Q, Psi : (d, d) positive definite matrices
    nu     : degrees of freedom (> d - 1)
    """
    d = Q.shape[0]
    sign_Q, logdet_Q = jnp.linalg.slogdet(Q)
    sign_Psi, logdet_Psi = jnp.linalg.slogdet(Psi)
    inv_Q = jnp.linalg.inv(Q)
    
    term1 = -0.5 * (nu + d + 1) * logdet_Q
    term2 = -0.5 * jnp.trace(Psi @ inv_Q)
    term3 = -0.5 * nu * d * jnp.log(2.0)
    term4 = -0.5 * logdet_Psi
    term5 = -log_multivariate_gamma(d, 0.5 * nu)

    return term1 + term2 + term3 + term4 + term5
