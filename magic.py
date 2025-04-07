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


def basis_convert(state: RBM_flexable):
    comb_state = state_preparation(state)
    N = state.in_features
    op_model = mcmc_optimize(comb_state, nk.hilbert.Qubit(2*N), 4, 2*N, 2**13)

    for i in range(N, 2*N):
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5)

    for i in range(N):
        op_model.model.control_z(i, i+N)

    for i in range(2*N):
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5)
    return op_model.model

# I -> |00> + |11> -> alpha = |00>
# Z -> |00> - |11> -> alpha = |10>
# X -> |01> + |10> -> alpha = |01>
# Y -> i(|01> - |10>) -> alpha = |11>


def commutator(alpha, beta):
    identity_bias_alpha = alpha[:, :alpha.shape[1] //
                                2] | alpha[:, alpha.shape[1] // 2:]
    identity_bias_beta = beta[:, :beta.shape[1] //
                              2] | beta[:, beta.shape[1] // 2:]
    temp = alpha ^ beta
    temp_merge = temp[:, :temp.shape[1] //
                      2] | temp[:, temp.shape[1] // 2:]
    temp_merge = temp_merge * identity_bias_alpha * identity_bias_beta
    ret = temp_merge.sum(axis=-1) % 2
    return ret


def bell_magic(state: RBM_flexable, n_chains: int, n_sweeps=10, n_samples=2**13, rndkey=1):
    nv = basis_convert(state)
    hi = nk.hilbert.Qubit(2*state.in_features)
    sampler_nv = nk.sampler.MetropolisLocal(
        hi, n_chains=n_chains, sweep_size=n_sweeps)
    sampler_nv_state = sampler_nv.init_state(nv, 1, jax.random.key(rndkey))
    samples_nv, sampler_nv_state = sampler_nv.sample(
        nv, 1, state=sampler_nv_state, chain_length=n_samples)
    samples_nv = samples_nv.reshape(4, -1, samples_nv.shape[-1])

    alpha = samples_nv[0] ^ samples_nv[1]
    beta = samples_nv[2] ^ samples_nv[3]

    commutator = commutator(alpha, beta)
    # in total n_chains * n_samples / 4 samples
    return 2 * commutator.sum() / (n_chains * n_samples / 4)
