"""
Random Walk Metropolis Hastings Class
"""

from abc import ABC, abstractmethod

import jax.random as jr
import jax.numpy as jnp
from jax import Array, vmap

from utils.dataset_utils import inverse_gamma
from feynman_kac.protocol import FeynmacKac


class FullConditional(ABC):
    """
    ABC protocol for FullConditional Samplers used by Particle Gibbs
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        Implement the full conditional sampling step here.

        params: dict containing all (required) params for the pGibbs step
        """


class MSVFactorLoadingsFC(FullConditional):
    """
    Gibbs sampling step for B
    """

    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        factors: (T, K) weighted avg of factor samples
        obs: (T, D)  returns Array
        """
        
        # load params
        factors = x_imm[-1]
        V = model.params["V"]
        D = V.shape[0]
        K = factors.shape[1] 
        sigma_b = 2  # hardcoded value for now

        # precalculate quantities
        F = factors.T @ factors
        FR_n = vmap(lambda R_i: F @ R_i)(obs.T)  # obs.T to vmap over stocks not time
        V_n = vmap(lambda vi: vi * jnp.eye(D) / sigma_b)(V)

        # sample new B
        keys = jr.split(key, D)
        B_new = vmap(
            lambda k, V_i, R_i, vi: jr.multivariate_normal(
                k,
                jnp.linalg.inv(F + V_i) @ factors @ R_i,
                vi * jnp.linalg.inv(F + V_i)
            )
        )(keys, V_n, FR_n, V)

        # mask out B to ensure identifiability
        rows = jnp.arange(D)[:, None]            # (D,1)
        cols = jnp.arange(K)[None, :]            # (1,K)
        mask = (cols <= rows).astype(B_new.dtype)  # (D,K): True for allowed entries
        B_new = B_new * mask

        key, _ = jr.split(keys[0])
        return key, B_new


class MSVFactorNoiseFC(FullConditional):
    """
    Gibbs sampling step for vi in MSV model
    """
    
    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        factors: (T, K) weighted avg of factor samples
        obs: (T, D)  returns Array
        """
        
        # load required params
        B = model.params["B"]
        D, K = B.shape
        alpha_v0, beta_v0 = 0.001, 0.001   # hard coded for now
        factors = x_imm[-1]
        T = factors.shape[0]

        # sample V
        keys = jr.split(key, D)
        V = vmap(
            lambda k, R_i, B_i: inverse_gamma(
                k,
                alpha=alpha_v0 + (T/2),
                beta=beta_v0 + 0.5 * jnp.linalg.norm(R_i - factors @ B_i)**2
            )(keys, obs.T, B)
        )

        key, _ = jr.split(keys[0])
        return key, V
    

class MSVEigenvalueNoiseFC(FullConditional):
    """
    Gibbs sampling step for sigma_h in MSV model
    """
    
    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        factors: (T, K) weighted avg of factor samples
        obs: (T, D)  returns Array
        """
        
        # load required params
        phi_h = model.params["phi_h"]
        h_0 = model.params["h_0"]
        h_iT = x_imm[0]
        T, K = h_iT.shape[0], h_iT.shape[1]
        alpha_0, beta_0 = 10, 0.1   # hard coded for now

        # precomputation
        c = 0.5 * jnp.sum(h_iT[1:] - h_0 - phi_h * (h_iT[:-1] - h_0))

        # sample sigma_h
        keys = jr.split(key, K)
        sigma_h_new = vmap(
            lambda k, phi_hi, h_i0: inverse_gamma(
                k,
                alpha_0 + (T/2),
                beta_0 + 0.5 * (1 - phi_hi**2) * (h_iT[0] - h_i0)**2 + c
            ) 
        )(keys, phi_h, h_0)

        key, _ = jr.split(keys[0])
        return key, sigma_h_new
    

class MSVRotationNoiseFC(FullConditional):
    """
    Gibbs sampling step for sigma_h in MSV model
    """
    
    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        factors: (T, K) weighted avg of factor samples
        obs: (T, D)  returns Array
        """
        
        # load required params
        phi_d = model.params["phi_d"]
        d_0 = model.params["d_0"]
        d_ijT = x_imm[1]
        T, K = d_ijT.shape[0], d_ijT.shape[1]
        alpha_0, beta_0 = 10, 0.1   # hard coded for now

        # precomputation
        c = 0.5 * jnp.sum(d_ijT[1:] - d_0 - phi_d * (d_ijT[:-1] - d_0), axis=2)

        # sample sigma_h
        keys = jr.split(key, K)
        sigma_h_new = vmap(
            lambda k, phi_hi, h_i0: inverse_gamma(
                k,
                alpha_0 + (T/2),
                beta_0 + 0.5 * (1 - phi_hi**2) * (d_ijT[0] - h_i0)**2 + c
            ) 
        )(keys, phi_d, d_0)

        key, _ = jr.split(keys[0])
        return key, sigma_h_new



def _sample_initial_hd(phi, x_iT, T, sigma):
    """
    Shared functionality for full conditional on h and delta
    """

    mean = (
        (1 - phi**2) * x_iT[0] + (1 - phi) * jnp.sum(x_iT[1:] - phi*x_iT[:-1])
    ) / (
        (1 - phi**2) + (T - 1) * (1 - phi)**2
    )
    var = (sigma**2) / (1 - phi**2 + (T - 1)*(1 - phi)**2)

    key, subkey = jr.split(key)
    x_i0_new = jr.multivariate_normal(subkey, mean, var)
    
    return key, x_i0_new

class MSVEigenvaluesFC(FullConditional):
    """
    h_i = log(lambda_i) eigenvalues of the decomp of Sigma_t
    This samples each h_i0 for each of the transformed eigV latent chains
    """
    
    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        """

        # load params
        phi_h = model.params["phi_h"]
        sigma_h = model.params["sigma_h"]
        h_iT = x_imm[0]
        T = h_iT.shape[0]

        return _sample_initial_hd(phi_h, h_iT, T, sigma_h)


class MSVRotationsFC(FullConditional):
    """
    d_ij = log(w_ij) on [-pi/2, pi/2]
    """
    
    def __init__(self, name: str):
        super().__init__(name)

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        """

        # load params
        phi_d = model.params["phi_d"]
        sigma_d = model.params["sigma_d"]
        d_ijT = x_imm[1]
        T, K = d_ijT.shape[:2]

        key, d_ijT_new = _sample_initial_hd(phi_d, d_ijT, T, sigma_d)

        # mask d_ijT
        mask_upper = jnp.triu(jnp.ones((K, K)), k=1)
        return key, d_ijT_new * mask_upper
