import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import vmap, Array, lax, jit
import jax.tree_util as jtu

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.datasets import Dataset


class LDSPrior:
    def __init__(self, T):
        self.T = T

    def sample(self, key, stationary_params, num_samples):
        m1 = stationary_params['m1']
        Q1 = stationary_params['Q1']
        A = stationary_params['A']
        b = stationary_params['b']
        Q = stationary_params['Q']

        As = jnp.tile(A[None], (self.T, 1, 1))
        bs = jnp.concatenate([
            m1[None],
            jnp.tile(b[None], (self.T-1, 1))
        ])
        Qs = jnp.concatenate([
            Q1[None],
            jnp.tile(Q[None], (self.T-1, 1, 1))
        ])

        @partial(jnp.vectorize, signature="(n),(t,d,d),(t,d),(t,d,d)->(t,d)")
        def sample_single(key, A, b, Q):
            biases = jr.multivariate_normal(key, b, Q)
            init_elems = (A, biases)

            @vmap
            def recursion(elem1, elem2):
                A1, b1 = elem1
                A2, b2 = elem2
                return A2 @ A1, A2 @ b1 + b2
            
            _, sample = jax.lax.associative_scan(recursion, init_elems)
            return sample

        keys = jr.split(key, num_samples)
        sample = sample_single(keys, As, bs, Qs)

        return sample

def random_rotation(key, n, theta=None):
    if n == 1:
        return jnp.eye(1) * jr.uniform(key)
    if theta is None:
        key, subkey = jr.split(key)
        theta = 0.5 * jnp.pi * jr.uniform(subkey)
    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                     [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def generate_linear_data(
    num_factors: int,
    latent_dim: int,
    emission_dim: int,
    num_sequences: int,
    num_timesteps: int,
    emission_cov: float,
    key: Array
) -> "Dataset":
    from utils.datasets import Dataset

    J, K, D, N, T = num_factors, latent_dim, emission_dim, num_sequences, num_timesteps

    key, key_m1, key_A, key_b, key_C, key_d = jr.split(key, 6)

    A = 0.95 * random_rotation(key_A, K, theta=jnp.pi/5) ### 0.95 ### theta
    Q = jnp.eye(K) - A @ A.T
    # Q = 0.1 * jnp.eye(K)

    params = {
        # 'm1': jr.normal(key_m1, shape=(K,)), # for now initialize both data and model at stationary distribution
        # 'Q1': Q,
        'm1': jnp.zeros(K),
        'Q1': jnp.eye(K),
        'A': A,
        # 'b': jr.normal(key_b, shape=(K,)),
        'b': jnp.zeros(K),
        'Q': Q,
        'R': jnp.eye(D) * jnp.sqrt(emission_cov) # keep R constant across J for now
    }

    lat_key, obs_key = jr.split(key)

    # sample an additional 25% sequences to use as validation data
    M = N + N // 4

    prior = LDSPrior(T)
    latent_sample = prior.sample(lat_key, params, M) # MxTxK

    emission_params = {
        'C': jr.normal(key_C, shape=(J, D, K)),
        'd': jr.normal(key_d, shape=(J, D)),
    }

    # not a pure function (uses M,T,latent_sample) but should be fine
    def sample_single_factor(emission_params, key):
        C, d, R = emission_params['C'], emission_params['d'], params['R']
        obs_sample = latent_sample @ C.T + jr.multivariate_normal(key, d, R, shape=(M, T)) # MxTxD
        return obs_sample

    obs_samples = vmap(sample_single_factor)(emission_params, jr.split(obs_key, J))

    emission_params = {
        "C": emission_params["C"][0],
        "d": emission_params["d"][0],
    }
    params.update(emission_params)
    
    return Dataset(
        train_data=tuple([o[:N] for o in obs_samples]),
        train_states=latent_sample[:N],
        val_data=tuple([v[N:] for v in obs_samples]),
        val_states=latent_sample[N:],
        params=params
    )


def givens_rotation(D, i, j, omega):
    """
    Create an n x n Givens rotation matrix that rotates in the (i, j) plane.

    Parameters:
        D (int): Dimension of the square matrix.
        i (int): First index (0-based).
        j (int): Second index (0-based).
        omega (float): Rotation angle in radians.

    Returns:
        jnp.ndarray: Givens rotation matrix.
    """
    
    # Start with identity
    G = jnp.eye(D)
    c = jnp.cos(omega)
    s = jnp.sin(omega)

    # Apply rotation in (i, j) plane
    G = G.at[i, i].set(c)
    G = G.at[j, j].set(c)
    G = G.at[i, j].set(s)
    G = G.at[j, i].set(-s)

    return G


@jit
def givens_product(w_ij: Array):
    """
    w_ij: (D, D) angles; we use only strict upper-triangle (i<j).
    
    Returns:
        G_all: (M, D, D) stack of Givens in lexicographic (i,j) order
        R: (D, D) product G_{(0)} @ G_{(1)} @ ... @ G_{(M-1)}
    """
    
    D = w_ij.shape[0]
    i_idx, j_idx = jnp.triu_indices(D, k=1)          # vectors of length M
    omegas = w_ij[i_idx, j_idx]                       # (M,)

    # construct Givens matrices over all (i,j,omega)
    G_all = vmap(
        lambda ii, jj, om: givens_rotation(D, ii, jj, om),
        in_axes=(0, 0, 0)
    )(i_idx, j_idx, omegas)

    # multiply all Givens matrices
    P = lax.associative_scan(lambda A, B: B @ A, G_all)[-1]
    return P


def invert_transform(delta):
    """ Invert transform delta to (-pi/2, pi/2) for Givens rotation angle """
    
    return (jnp.pi / 2) * (jnp.exp(delta) - 1) / (jnp.exp(delta) + 1)


def inverse_gamma(key, alpha, beta, shape):
    return 1.0 / (jr.gamma(key, alpha, shape) / beta) 


def generate_stock_price_data(key: jr.PRNGKey, K: int, D: int, T: int):
    """
    Generate returns according to the model in scalable inference paper

    K: factor dimension
    D: returns dimension
    """
    from utils.datasets import Dataset

    assert D >= K, "Factor dimension cannot be greater than observation dimension"

    # randomly sample all necessary parameters 
    key, key_h = jr.split(key)
    h_i0 = jr.normal(key_h, shape=(K))

    key, key_d = jr.split(key)
    d_ij0 = jr.normal(key_d, shape=(K, K))
    d_ij0 = d_ij0.at[jnp.tril_indices(K)].set(0)

    key, key_lambda_h, key_mu_h, key_phi_h = jr.split(key, 4)
    lambda_h = jr.gamma(key_lambda_h, 1)
    mu_h = jr.normal(key_mu_h) * (1.0 / jnp.sqrt(lambda_h))
    phi_h_hat = mu_h + (jr.normal(key_phi_h, shape=(K)) * jnp.sqrt(1.0 / lambda_h))
    phi_h = (jnp.exp(phi_h_hat) - 1) / (jnp.exp(phi_h_hat) + 1)
    
    key, key_lambda_d, key_mu_d, key_phi_d = jr.split(key, 4)
    lambda_d = jr.gamma(key_lambda_d, 1)
    mu_d = jr.normal(key_mu_d) * (1.0 / jnp.sqrt(lambda_d))
    phi_d_hat = mu_d + (jr.normal(key_phi_d, shape=(K, K)) * jnp.sqrt(1.0 / lambda_d))
    phi_d = (jnp.exp(phi_d_hat) - 1) / (jnp.exp(phi_d_hat) + 1)

    key, key_sigma_h = jr.split(key)
    sigma_h = jnp.sqrt(inverse_gamma(key_sigma_h, 10, 0.1, (K)))

    key, key_sigma_d = jr.split(key)
    sigma_d = jnp.sqrt(inverse_gamma(key_sigma_d, 10, 0.1, (K, K)))
    
    key, key_h1 = jr.split(key)
    # h_i1 = (jr.normal(key_h1, shape=(K)) + h_i0) * sigma_h / jnp.sqrt(1 - phi_h**2)
    h_i1 = h_i0 + (sigma_h / jnp.sqrt(1.0 - phi_h**2)) * jr.normal(key_h1, shape=(K))

    key, key_d1 = jr.split(key)
    # d_ij1 = (jr.normal(key_d1, shape=(K, K)) + d_ij0) * sigma_d / jnp.sqrt(1 - phi_d**2)
    d_ij1 = d_ij0 + (sigma_d / jnp.sqrt(1.0 - phi_d**2)) * jr.normal(key_d1, shape=(K, K))
    d_ij1 = d_ij1 * jnp.triu(jnp.ones((K, K), dtype=d_ij1.dtype), k=1)  # keep only i<j
    
    key, key_B = jr.split(key)
    B = jr.normal(key_B, shape=(D, K)) if D != K else jnp.eye(D)
    rows = jnp.arange(D)[:, None]            # (D,1)
    cols = jnp.arange(K)[None, :]            # (1,K)
    mask = (cols <= rows).astype(B.dtype)  # (D,K): True for allowed entries
    B = B * mask

# Set the first min(D,K) diagonal entries to 1 (identification)
    m = jnp.minimum(D, K)
    B = B.at[jnp.arange(m), jnp.arange(m)].set(1.0)

    key, key_V = jr.split(key)
    V = jnp.diag(jr.uniform(key_V, shape=(D)))

    # calculate factors and returns at t1 
    w_ij1 = invert_transform(d_ij1)  # (jnp.pi / 2) * (jnp.exp(d_ij1) - 1) / (jnp.exp(d_ij1) + 1)
    P1 = givens_product(w_ij1)
    L1 = jnp.diag(jnp.exp(h_i1))
    S1 = P1 @ L1 @ P1.T

    key, key_f, key_r = jr.split(key, 3)
    f1 = jr.multivariate_normal(key_f, jnp.zeros(K), S1)
    r1 = jr.multivariate_normal(key_r, B @ f1, V)

    # pre-generate noise for latent Gaussian processes
    key, key_eta_h, key_eta_d = jr.split(key, 3)
    eta_h = jr.normal(key_eta_h, shape=(T-1, K))
    eta_d = jr.normal(key_eta_d, shape=(T-1, K, K))

    # simulate latent dynamics
    mask_upper = jnp.triu(jnp.ones((K, K)), k=1)
    def step(carry, noise):
        
        h_it, d_ijt = carry   # other params are accessed from global scope
        eta_ht, eta_dt = noise
        
        h_i = h_i0 + phi_h * (h_it - h_i0) + sigma_h * eta_ht 
        d_ij = d_ij0 + phi_d * (d_ijt - d_ij0) + sigma_d * eta_dt
        d_ij = d_ij * mask_upper

        return (h_i, d_ij), (h_i, d_ij)
        
    carry0 = (h_i1, d_ij1)
    noise = (eta_h, eta_d)

    _, paths = lax.scan(step, carry0, noise)

    # back-calculate all rotation angles, factors and returns
    w_ij = vmap(lambda delta: invert_transform(delta))(paths[1])   # (T, K, K)
    P = vmap(lambda omega: givens_product(omega))(w_ij)            # (T, K, K)
    L = vmap(lambda h: jnp.diag(jnp.exp(h)))(paths[0])             # (T, K, K)
    S = vmap(lambda Pt, Lt: Pt @ Lt @ Pt.T)(P, L)

    keys = jr.split(key, 2*(T-1) + 1)
    f = vmap(lambda k, St: jr.multivariate_normal(k, jnp.zeros(K), St))(keys[:T-1], S)
    r = vmap(lambda k, ft: jr.multivariate_normal(k, B @ ft, V))(keys[T-1:-1], f)

    paths = jtu.tree_map(
        lambda x0, xT: jnp.concatenate([x0[None, ...], xT], axis=0),
        carry0, paths
    )      
    f = jnp.concatenate([f1[None, ...], f])
    r = jnp.concatenate([r1[None, ...], r])

    params = {
        "h0": h_i0,
        "d0": d_ij0,
        "phi_h": phi_h,
        "phi_d": phi_d,
        "sigma_h": sigma_h,
        "sigma_d": sigma_d,
        "B": B,
        "V": V
    }
    
    return Dataset(
        train_data=r,
        train_states=(paths, f),
        val_data=None,
        val_states=None,
        params=params
    )