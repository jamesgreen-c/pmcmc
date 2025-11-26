from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jr
from jax import Array, tree_map, jit

from feynman_kac.protocol import PFOutputs
from feynman_kac.bootstrap import BaseParticleFilter

from resample.backward_sampling import BaseBackwardSampler


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

    def __init__(self, pf: BaseParticleFilter, cfg: PGConfig, backward_sampler: BaseBackwardSampler):
        self.pf = pf
        self.cfg = cfg
        self.key = cfg.key
        self.backward_sampler = backward_sampler.sample

    @abstractmethod
    def update_params(
        self, 
        key: jr.PRNGKey, 
        curr_params: dict, 
        outs: PFOutputs, 
        obs: Array, 
        ref
    ):
        """
        Implement the model specific parameter update steps here.
                
        :param key: jr.PRNGKey
        :param curr_params: Current model parameter values
        :param outs: PFOutputs
        :param obs: Array of observations
        :param ref: Reference trajectory
        """
        pass

    def run(self, obs: Array, ref: Array | None = None, params0: dict | None = None):
        """
        Run particle Gibbs

        :param obs: Array of observations
        :param ref: Reference trajectory
        :param params0: Initial model parameters
        """

        params = params0 if params0 is not None else self.pf.model.params
        outs = None

        params_chain, paths_chain, logZs, ess = [], [], [], []
        pbar = tqdm(range(self.cfg.n_iters))
        for iter in pbar: 
            
            self.key, key1, key2 = jr.split(self.key, 3)
            
            ref, outs = self.update_states(key1, params, obs, ref)
            params = self.update_params(key2, params, outs, obs, ref)

            params_chain.append(params)
            paths_chain.append(ref)
            logZs.append(outs.logZ_hat)
            ess.append(outs.ess_history)

            pbar.set_postfix({"log likelihood": float(outs.logZ_hat)})

        thetas = tree_map(lambda *xs: jnp.stack(xs), *params_chain)
        x_paths = jnp.stack(paths_chain)     # (n_iters, T, d)
        logZs   = jnp.stack(logZs)           # (n_iters,)
        ess_hist= jnp.stack(ess)             # (n_iters, T)
        return PGSamples(thetas, x_paths, logZs, ess_hist), outs

    def update_states(
        self, 
        key: jr.PRNGKey, 
        params: dict, 
        obs: Array, 
        ref: Array | None = None
    ):
        """
        Update the reference path via conditional SMC and backward sampling. 

        :param key: jr.PRNGKey
        :param params: Model parameters
        :param obs: Array of observations
        :param ref: Current reference trajectory
        """        

        self.pf.update_params(params)
        outs: PFOutputs = self.pf.filter(key, obs, ref)

        # backward sampling
        key, subkey = jr.split(key)
        ref_new = self.backward_sampler(
            subkey, 
            self.pf, 
            outs.particles, 
            outs.ancestors, 
            outs.weights
        )
        
        return ref_new, outs
