from typing import Any
from dataclasses import dataclass
from functools import partial
from abc import ABC, abstractmethod
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap, tree_map, lax, jit

from feynmac_kac.protocol import CSMC, PFOutputs
from feynmac_kac.utils import log_normalize
from resample.resamplers import single_multinomial

import time

@dataclass
class PGConfig:
    n_iters: int
    T: int
    key: jr.PRNGKey
    backward_sample: bool = True

@dataclass
class PGSamples:
    thetas: Any                  # stacked theta pytree 
    x_paths: Array               # (n_iters, T, d)
    logZs: Array                 # (n_iters,)
    ess_histories: Array         # (n_iters, T)


class ParticleGibbs(ABC):
    """
    This is a generic protocol for Particle Gibbs type models
    """

    def __init__(self, csmc: CSMC, cfg: PGConfig):
        self.csmc = csmc
        self.jit_csmc = jit(self.csmc.csmc)
        self.jit_backward_sampling = jit(self.backward_sampling)
        self.jit_extract_trajectory = jit(self.extract_one_trajectory)
        self.cfg = cfg
        self.key = cfg.key

    @abstractmethod
    def update_params(self, key: jr.PRNGKey, curr_params: dict, outs: PFOutputs, obs: Array, x_imm):
        pass

    def run(self, obs: Array, x_imm: Array | None = None, params0: dict | None = None):
        """
        Run particle Gibbs
        """

        params = params0 if params0 is not None else self.csmc.model.params
        outs = None

        params_chain, paths_chain, logZs, ess = [], [], [], []
        pbar = tqdm(range(self.cfg.n_iters))
        for iter in pbar: 
            
            self.key, key1, key2 = jr.split(self.key, 3)
            
            # start = time.perf_counter()
            x_imm, outs = self.update_states(key1, params, obs, x_imm)
            # print(f"Update states took: {time.perf_counter() - start:.2f} seconds")
            
            # start = time.perf_counter()
            params = self.update_params(key2, params, outs, obs, x_imm)
            # print(f"Update params took: {time.perf_counter() - start:.2f} seconds")

            params_chain.append(params)
            paths_chain.append(x_imm)
            logZs.append(outs.logZ_hat)
            ess.append(outs.ess_history)

            pbar.set_postfix({"log likelihood": float(outs.logZ_hat)})

        thetas = tree_map(lambda *xs: jnp.stack(xs), *params_chain)
        x_paths = jnp.stack(paths_chain)     # (n_iters, T, d)
        logZs   = jnp.stack(logZs)           # (n_iters,)
        ess_hist= jnp.stack(ess)             # (n_iters, T)
        return PGSamples(thetas, x_paths, logZs, ess_hist), outs

    def update_states(self, key: jr.PRNGKey, params: dict, obs: Array, x_imm: Array | None = None):
        
        # print(params)
        self.csmc.model.update(params)
        outs: PFOutputs = self.jit_csmc(key, obs, x_imm)

        # optional backward sampling
        key, subkey = jr.split(key)
        # start = time.perf_counter()
        if self.cfg.backward_sample:
            new_x = self.jit_backward_sampling(subkey, outs.particles, outs.ancestors, outs.weights)
        else:
            new_x = self.jit_extract_trajectory(subkey, outs.particles, outs.ancestors, outs.weights)
        # print(f"Backward sampling time: {time.perf_counter() - start:.2f} seconds")
        return new_x, outs

    def backward_sampling(self, key: jr.PRNGKey, particles: Array, ancestors: Array, weights: Array):
        """
        Perform backward sampling on the given particles and weights.
        """

        T = particles.shape[0]
        B = jnp.zeros((T,), dtype=jnp.int32)
        B = B.at[T - 1].set(jnp.squeeze(single_multinomial(key, weights[-1])))

        log_pt = vmap(lambda t, x_t, x_prev: self.csmc.model.log_pt(t, x_t, x_prev), in_axes=(None, None, 0))
        
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

        # for t in reversed(range(particles.shape[0] - 1)):
        #     key, subkey = jr.split(key)
        #     log_W_tilde = jnp.log(weights[t] + 1e-30) + log_pt(t + 1, particles[t + 1, B[t + 1]], particles[t])
        #     W_tilde = log_normalize(log_W_tilde)[0]
        #     B = B.at[t].set(single_multinomial(subkey, W_tilde).item())
        
        # return particles[jnp.arange(T), B]

    
    def extract_one_trajectory(
        self,
        key: jr.PRNGKey, 
        particles: Array, 
        ancestors: Array, 
        weights: Array
    ) -> Array:
        """
        Extract a single trajectory from the particle history.
        The final state is chosen randomly, then the corresponding trajectory
        is constructed backwards, until time t=0.

        particles: (T, N, D)
        weights: (T, N)
        ancestors: (T, N)
        """

        T, N, D = particles.shape
        x_imm = jnp.zeros((T, D))  # dummy vector
        
        # sample B_T then walk along ancestry
        n = single_multinomial(key, weights[-1])
        for t in reversed(range(T)):
            x_imm = x_imm.at[t].set(particles[t, n])
            if t > 0:
                n = ancestors[t, n]
        return x_imm
