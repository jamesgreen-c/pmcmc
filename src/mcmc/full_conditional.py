"""
Random Walk Metropolis Hastings Class
"""

from abc import ABC, abstractmethod
from jax import Array
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
        pass
