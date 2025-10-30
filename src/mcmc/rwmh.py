"""
Random Walk Metropolis Hastings Class
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

import jax.random as jr
import jax.numpy as jnp
from jax import Array, vmap
from jax.scipy.stats import norm

from feynmac_kac.protocol import FeynmacKac
from mcmc.utils import invwishart_logpdf


@dataclass
class RWMHConfig:
    name: str
    scale: float
    n_burn: int
    adaptive: bool = False
    


class RandomWalkMH(ABC):
    """
    
    curr: the current parameter value
    """

    def __init__(self, cfg: RWMHConfig):

        self.cfg = cfg
        self.robbins = 0  # Robbins-Monro Iterations
        self.accept_rate = 0.0

    @abstractmethod
    def log_prior(self, param: Array):
        """ Implement a method to evaluate the log prior of the parameter"""
        pass

    def log_p(self, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        Implement a method to evaluate the log probability of your model.
        """

        # don't mutate any of the original models params
        model = model.copy()
        model.update({self.cfg.name: curr})

        # model is Feynman Kac so always has log_p0 and log_pt
        log_p0 = model.log_p0(x_imm[0])
        log_p_all_t = vmap(model.log_pt, in_axes=(0, 0, 0))(
            jnp.arange(1, x_imm.shape[0]), 
            x_imm[1:],
            x_imm[:-1]
        )
        log_f_all_t = vmap(model.log_f, in_axes=(0, 0, 0))(
            jnp.arange(0, x_imm.shape[0]),
            x_imm,
            obs
        )
        log_prior = self.log_prior(curr)

        logp = log_p0 + jnp.sum(log_p_all_t) + jnp.sum(log_f_all_t) + log_prior
        return logp

    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """
        
        Log transform preserves inequality
        """
        
        # propose
        key, subkey = jr.split(key)
        proposal = self.propose(subkey, curr)
        # print(self.cfg.name, " curr shape: ", curr.shape)
        # print(self.cfg.name, " proposal shape: ", proposal.shape)

        # calculate acceptance prob
        logp_curr = self.log_p(curr, model, x_imm, obs)
        logp_prop = self.log_p(proposal, model, x_imm, obs)
        log_r = logp_prop - logp_curr

        # accept or reject
        key, subkey = jr.split(key)
        u = jnp.log(jr.uniform(subkey))  # default [0, 1] 
        accept = u < log_r

        new = jnp.where(accept, proposal, curr)
        # print(self.cfg.name, " new shape: ", new.shape)
        self.accept_rate = 0.9 * self.accept_rate + 0.1 * accept  # track acceptance rate

        # adapt scale
        if self.cfg.adaptive and self.robbins < self.cfg.n_burn:
            self.robbins += 1
            self.adapt_scale(accept)

        return key, new

    def propose(self, key, curr: Array):
        """ 
        Propose a new parameter value
        Permit arrays (eg mean may be vector valued)

        Returns the proposal, and Jacobian if required
        """
        return curr + self.cfg.scale * jr.normal(key, shape=curr.shape)
    
    def adapt_scale(self, accept: bool, target_rate = 0.234):
        """
        Robbins-Monro style adaptation of the proposal scale.
        """

        # decaying step size: 1/sqrt(t) keeps ergodicity
        gamma_t = 1.0 / jnp.sqrt(self.robbins + 1)

        # move log(scale) toward target accept rate
        log_scale = jnp.log(self.cfg.scale)
        log_scale += gamma_t * (accept - target_rate)
        self.cfg.scale = jnp.maximum(jnp.exp(log_scale), 1e-8)


class CholeskyRWMH(RandomWalkMH, ABC):

    def __init__(self, cfg: RWMHConfig):
        super().__init__(cfg)

    # override sample
    def sample(self, key, curr: Array, model: FeynmacKac, x_imm: Array, obs: Array):
        """

        Log transform preserves inequality
        """
        
        # propose
        key, subkey = jr.split(key)
        proposal, L, L_prop = self.propose(subkey, curr)
        
        # calculate log jacobians
        log_jac_curr = self.log_jac(L)
        log_jac_prop = self.log_jac(L_prop)

        # calculate acceptance prob
        logp_curr = self.log_p(curr, model, x_imm, obs) + log_jac_curr
        logp_prop = self.log_p(proposal, model, x_imm, obs) + log_jac_prop
        log_r = logp_prop - logp_curr

        # accept or reject
        key, subkey = jr.split(key)
        u = jnp.log(jr.uniform(subkey))  # default [0, 1] 
        accept = u < log_r

        new = jnp.where(accept, proposal, curr)
        # print(self.cfg.name, " new shape: ", new.shape)
        self.accept_rate = 0.9 * self.accept_rate + 0.1 * accept  # track acceptance rate

        # adapt scale
        if self.cfg.adaptive and self.robbins < self.cfg.n_burn:
            self.robbins += 1
            self.adapt_scale(accept)

        return key, new

    # override propose
    def propose(self, key, curr: Array):
        """
        Propose a new parameter value
        Should always be a matrix

        We use Cholesky Decomp to ensure PSD of matrices
        Returns the proposal, and Jacobian if required
        """

        assert curr.ndim == 2 and curr.shape[0] == curr.shape[1], "Parameter must be square matrix"
        
        d = curr.shape[0]
        L = jnp.linalg.cholesky(0.5*(curr + curr.T) + 1e-6*jnp.eye(d))
        L_prop = L + self.cfg.scale * jr.normal(key, L.shape)
        
        # enforce positive diag for the Jacobian formula
        diag = jnp.abs(jnp.diag(L_prop)) + 1e-8
        L_prop = L_prop.at[jnp.diag_indices(d)].set(diag)
        Q_prop = L_prop @ L_prop.T
        
        return Q_prop, L, L_prop
    
    def log_jac(self, L: Array):
        d = L.shape[0]
        exponents = jnp.arange(d, 0, -1)           # [d, d-1, ..., 1]
        return d * jnp.log(2.0) + jnp.dot(exponents, jnp.log(jnp.diag(L) + 1e-12))


class MatrixRWMH(RandomWalkMH):

    def __init__(self, cfg, means: Array, sigma: float):
        super().__init__(cfg)
        self.means = means
        self.sigma = sigma
    
    def log_prior(self, param: Array):

        # double vmap over indices
        log_prior_ij = vmap(
            lambda i: vmap(
                lambda j: norm.logpdf(param[i, j], loc=self.means[i, j], scale=self.sigma)
            )(jnp.arange(self.means.shape[1]))
        )(jnp.arange(self.means.shape[0]))
        
        return jnp.sum(log_prior_ij)
    

class CovarianceRWMH(CholeskyRWMH):
    
    def __init__(self, cfg: RWMHConfig, psi: Array | float, nu: float):
        super().__init__(cfg)
        self.psi = psi
        self.nu = nu 
    
    def log_prior(self, param: Array):
        Psi = self.psi * jnp.eye(param.shape[0]) if isinstance(self.psi, float) else self.psi 
        return invwishart_logpdf(param, Psi, self.nu)
