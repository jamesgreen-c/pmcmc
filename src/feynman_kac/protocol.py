from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol

import copy

import jax.numpy as jnp
import jax.random as jr
from jax import Array, tree_util


class FeynmacKac(ABC):
    """
    This is a generic protocol for Feynman-Kac type models.
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def p0(self, key: jr.PRNGKey, N: int) -> Array:
        """
        Define the initial distribution of particles.
        """
        pass

    @abstractmethod
    def pt(self, key: jr.PRNGKey, x: Array, t: int) -> Array:
        """
        Define the transition kernel of the particles.

        x: (d, )
        """
        pass 

    # def g(self, t: int, x_t: Array, x_prev: Array | None = None, y_t: Array | None = None) -> Array:
    #     """
    #     Define the potential function at time t.

    #     x_t, x_prev: (d, )
    #     y_t: (k, )

    #     """

    @abstractmethod
    def log_g(self, t: int, x_t: Array, x_prev: Array | None = None, y_t: Array | None = None) -> Array:
        """
        Define the log-potential function at time t.

        x_t, x_prev: (d, )
        y_t: (k, )
        """
        pass

    @abstractmethod
    def log_f(self, t: int, x_t: Array, y_t: Array):
        """ 
        Implement the emission probability only.
        This is for chain likelihood calculations when Guided PFs are used such 
        that the potential function is no longer equal to the emission.
        """

    @abstractmethod
    def log_pt(self, t: int,  x_t: Array, x_prev: Array) -> Array:
        """
        Evaluate the log-transition density at time t.
        x_t, x_prev: (d, )
        """
        pass

    @abstractmethod
    def log_p0(self, x_0: Array):
        """ 
        Define method to evaluate the t=0 probability
        """

    def update(self, params: dict):
        """
        Update model parameters.
        """
        new_params = {**self.params, **params}
        self.params = new_params

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class PFConfig:
    N: int
    key: jr.PRNGKey
    resample_scheme: str = "systematic"   # 'multinomial' | 'systematic' | 'stratified' | 'residual'
    ess_threshold: float | None = 0.5     # trigger when ESS/N < threshold; None disables


@tree_util.register_pytree_node_class
@dataclass
class PFOutputs:
    particles: Array
    weights: Array
    ancestors: Array
    logZ_hat: Array
    ess_history: Array

    def tree_flatten(self):
        children = (self.particles, self.weights, self.ancestors, self.logZ_hat, self.ess_history)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        particles, weights, ancestors, logZ_hat, ess_history = children
        return cls(particles, weights, ancestors, logZ_hat, ess_history)

# class CSMC(Protocol):
    
#     model: FeynmacKac

#     def filter(self, T: int, obs: Array) -> PFOutputs:
#         """
#         """
        
#     def csmc(self, obs: Array, x_imm: Array | None) -> PFOutputs:
#         """
#         """