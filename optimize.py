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

    def stochastic_reconfiguration_H(self, qubit, tol=1e-3, lookback=5, max_iters=1000, lr=1e-1, lr_tau=None, lr_min=0.0, eps=1e-4):
        target_model = RBM_H_State(self.model, qubit)
