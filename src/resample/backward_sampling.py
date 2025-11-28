from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, lax, jit
import jax.tree_util as jtu

from feynman_kac.protocol import FeynmacKac, PFConfig, PFOutputs
from feynman_kac.bootstrap import BaseParticleFilter

from resample.resamplers import RESAMPLERS, Resampler, single_multinomial, multinomial_resample
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


class SmoothingDistribution(BaseBackwardSampler):
    """
    Implements the Backward Simulation Particle Smoother (BSPS) of Särkkä (2013).

    This is the standard backward resampling smoother described in:
        S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013.
        (See Chapter 10, especially Section 10.4: Backward Simulation Smoothing.)

    Given the output of a particle filter — particles, normalized weights, and
    ancestor indices — this algorithm:
        • draws the final state index according to the filtering weights,
        • moves backward in time sampling ancestor indices according to
          the backward transition probabilities proportional to:
                w_t(i) ⋅ p(x_{t+1}^k | x_t^i),
        • returns both the sampled trajectories and their empirical mean as
          a particle approximation to the smoothing distribution p(x_t | y_{1:T}).

    The implementation follows the structure of BSPS in Särkkä (2013)
    """

    # TODO permit a sequential version that avoids computing the full N x N matrix

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

        T, N = particles.shape[0], particles.shape[1]
        B = jnp.zeros((T,N), dtype=jnp.int32)
        B = B.at[T - 1].set(multinomial_resample(key, weights[-1]))

        vmap_log_pt = vmap(
            vmap(
                lambda t, x_t, x_prev: model.log_pt(t, x_t, x_prev), 
                in_axes=(None, None, 0)
            ),
            in_axes=(None, 0, None)
        )
        vmap_log_normalize = vmap(log_normalize)
        vmap_single_multinomial = vmap(single_multinomial, in_axes=(0, 0))
        
        def step(carry, t):
            key, B, _ = carry
            key, subkey = jr.split(key)
            idx = T - 1 - t  # reverse index
            
            # sample ancestors``
            x_next = particles[idx + 1, B[idx + 1]]
            x_nt = particles[idx]
            log_W_tilde = jnp.log(weights[idx] + 1e-30)[None, :] + vmap_log_pt(idx + 1, x_next, x_nt)  # (N, N)
            W_tilde, _ = vmap_log_normalize(log_W_tilde)  # (N, N)

            subkeys = jr.split(subkey, N)
            B = B.at[idx].set(jnp.squeeze(vmap_single_multinomial(subkeys, W_tilde)))

            return (key, B, _), None

        carry = (key, B, weights[-1])
        (key, B, _), _ = lax.scan(step, carry, jnp.arange(1, T))
        
        # calculate smoothing approximation as the empirical distribution of particles
        backward_samples = particles[jnp.arange(T)[:, None], B]
        x_hat = backward_samples.mean(axis=1)
        return x_hat, backward_samples