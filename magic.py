import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import netket as nk
import matplotlib.pyplot as plt
import copy
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.linalg import solve, block_diag

from rbm import RBM_flexable, log_wf_grad, RBM_H_State, state_preparation
from optimize import mcmc_optimize


def bell_magic(state: RBM_flexable):
    comb_state = state_preparation(state)
    N = state.in_features
    op_model = mcmc_optimize(comb_state, nk.hilbert.Qubit(2*N), 4, 2*N, 2**13)

    for id in range(N, N+1):
        op_model.stochastic_reconfiguration_H(
            id, resample_phi=5, outprintconfig=5)

    return op_model.model
    # for id in range(N):
    #     op_model.model.control_z(id, id+N)
    # for id in range(2*N):
    #     op_model.stochastic_reconfiguration_H(
    #         id, resample_phi=5, outprintconfig=1)
    # return op_model.model
