from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, lax, jit
import jax.tree_util as jtu

from feynman_kac.protocol import FeynmacKac, PFConfig, PFOutputs
from feynman_kac.bootstrap import BaseParticleFilter

from resample.resamplers import RESAMPLERS, Resampler, single_multinomial
from feynman_kac.utils import log_normalize, ess


class BaseBackwardSampler(ABC):
    """
    Strategy interface for sampling a single latent trajectory from
    the outputs of a particle filter (particles, ancestors, weights).

    Implementations may include:
        - Standard backward simulation (Godsill et al.)
        - FFBS (Forward-Filter Backward-Sampler)
        - Deterministic extraction of MAP particle
    """

    @abstractmethod
    def sample(cls, key, model, particles, ancestors, weights):
        """
        Implement the desired backward sampling strategy here.

        key: PRNGKey
        model: FeynmacKac model according to protocol
        particles: (T, N, d) array of particles from PF
        ancestors: (T - 1, N) array of ancestor indices from PF
        weights: (T, N) array of normalized weights from PF
        """
        pass


class BackwardSampler(BaseBackwardSampler):
    """
    Implements the standard backward-resampling algorithm used in
    Particle Gibbs and conditional SMC.

    Uses model.log_pt to compute backward transition probabilities and
    performs multinomial resampling at each time step.
    """

    @classmethod
    def sample(
        cls, 
        key: jr.PRNGKey, 
        model: FeynmacKac, 
        particles: Array, 
        ancestors: Array, 
        weights: Array
    ):
        """
        Perform standard backward sampling on the given particles and weights.

        key: PRNGKey
        model: FeynmacKac model according to protocol
        particles: (T, N, d) array of particles from PF
        ancestors: (T - 1, N) array of ancestor indices from PF
        weights: (T, N) array of normalized weights from PF
        """

        T = particles.shape[0]
        B = jnp.zeros((T,), dtype=jnp.int32)
        B = B.at[T - 1].set(jnp.squeeze(single_multinomial(key, weights[-1])))

        log_pt = vmap(lambda t, x_t, x_prev: model.log_pt(t, x_t, x_prev), in_axes=(None, None, 0))
        
        def step(carry, t):
            key, B = carry
            key, subkey = jr.split(key)
            idx = T - 1 - t  # reverse index
            
            # sample ancestors
            x_next = particles[idx + 1, B[idx + 1]]
            x_nt = particles[idx]
            log_W_tilde = jnp.log(weights[idx] + 1e-30) + log_pt(idx + 1, x_next, x_nt)
            W_tilde, _ = log_normalize(log_W_tilde)
            B = B.at[idx].set(jnp.squeeze(single_multinomial(subkey, W_tilde)))
            return (key, B), None

        carry = (key, B)
        (key, B), _ = lax.scan(step, carry, jnp.arange(1, T))
        return particles[jnp.arange(T), B]


class LineageTracking(BaseBackwardSampler):
    """
    Implements deterministic extraction of a single trajectory
    from the particle filter outputs by following the lineage
    of a sampled final particle.

    This method does not require model.log_pt.
    """

    @classmethod
    def sample(
        cls, 
        key: jr.PRNGKey, 
        model: FeynmacKac, 
        particles: Array, 
        ancestors: Array, 
        weights: Array
    ):
        """
        Extract a single trajectory by following the lineage
        of a sampled final particle.

        key: PRNGKey
        model: FeynmacKac model according to protocol
        particles: (T, N, d) array of particles from PF
        ancestors: (T - 1, N) array of ancestor indices from PF
        weights: (T, N) array of normalized weights from PF
        """

        T = particles.shape[0]
        B = jnp.zeros((T,), dtype=jnp.int32)
        B = B.at[T - 1].set(jnp.squeeze(single_multinomial(key, weights[-1])))

        def step(carry, t):
            B = carry
            idx = T - 1 - t  # reverse index
            B = B.at[idx].set(ancestors[idx, B[idx + 1]])
            return B, None

        B, _ = lax.scan(step, B, jnp.arange(1, T))
        return particles[jnp.arange(T), B]
    