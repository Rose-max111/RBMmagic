import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm

from rbm import RBM_flexable, log_wf_grad, RBM_H_State


class mcmc_optimize():
    def __init__(self, model: RBM_flexable, hilbert: nk.hilbert.Hilbert, n_chains: int, n_sweeps: int, n_samples: int, n_discard: int):
        self.model = model
        self.hilbert = hilbert
        self.n_chains = n_chains
        self.n_sweeps = n_sweeps
        self.n_samples = n_samples
        self.n_discard = n_discard

    def stochastic_reconfiguration_H(self, qubit, tol=1e-3, lookback=5, max_iters=1000, resample_phi=None, lr=1e-1, lr_tau=None, lr_min=0.0, eps=1e-4, rndkey_phi=0, rndkey_psi=1):
        # init target model distribution and its sampler
        target_model = RBM_H_State(self.model, qubit)
        sampler_phi = nk.sampler.MetropolisLocal(
            self.hilbert, n_chains=self.n_chains, sweep_size=self.n_sweeps)
        sampler_phi_state = sampler_phi.init_state(
            target_model, 1, jax.random.key(rndkey_phi))
        # sample from the target model distribution
        samples_phi, sampler_phi_state = sampler_phi.sample(
            target_model, 1, state=sampler_phi_state, chain_length=self.n_samples)
