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
import timeit

from rbm import RBM_flexable, log_wf_grad, RBM_H_State, state_preparation
from optimize import mcmc_optimize


def exact_fidelity(psi, phi):
    return np.abs(np.sum(psi.conj()*phi)) / np.linalg.norm(psi) / np.linalg.norm(phi)


def array2num(arr):
    return np.dot(arr, 2**np.arange(arr.size)[::-1])


def num2array(num, N):
    arr = np.zeros(N, dtype=int)
    for i in range(N):
        arr[i] = (num >> (N - i - 1)) & 1
    return arr


def hadamard(amplitude, qubit):
    N = int(np.log2(amplitude.shape[0]))
    new_amplitude = np.zeros(amplitude.shape, dtype=amplitude.dtype)
    for i in range(amplitude.shape[0]):
        arr = num2array(i, N)
        new_amplitude[i] += 1/np.sqrt(2) * \
            amplitude[i] * (-1)**arr[qubit]
        arr[qubit] = 1 - arr[qubit]
        new_amplitude[array2num(arr)] += 1/np.sqrt(2) * \
            amplitude[i]
    return new_amplitude


def control_z(amplitude, qubit1, qubit2):
    N = int(np.log2(amplitude.shape[0]))
    new_amplitude = np.zeros(amplitude.shape, dtype=amplitude.dtype)
    for i in range(amplitude.shape[0]):
        arr = num2array(i, N)
        new_amplitude[i] += amplitude[i] * (-1)**(arr[qubit1] * arr[qubit2])
    return new_amplitude


def basis_convert(state: RBM_flexable, compairing_exact_state=None):
    comb_state = state_preparation(state)
    N = state.in_features

    if compairing_exact_state is not None:
        hi = nk.hilbert.Qubit(2*N)
        all_state = hi.all_states()
        compairing_exact_state = np.exp(comb_state(all_state))

    op_model = mcmc_optimize(comb_state, nk.hilbert.Qubit(2*N), 16, 2*N, 2**13)
    overlap_history = []
    print("Step One")
    for i in range(N, 2*N):
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5, lr_tau=0.5, lr_nstep=30, lr_min=1e-2)
        if compairing_exact_state is not None:
            compairing_exact_state = hadamard(compairing_exact_state, i)
            approx_amplitude = np.exp(op_model.model(all_state))
            overlap_history.append(exact_fidelity(
                compairing_exact_state, approx_amplitude))
            print(overlap_history[-1])

    print("Step Two")
    for i in range(N):
        op_model.model.control_z(i, i+N)
        if compairing_exact_state is not None:
            compairing_exact_state = control_z(compairing_exact_state, i, i+N)
            approx_amplitude = np.exp(op_model.model(all_state))
            overlap_history.append(exact_fidelity(
                compairing_exact_state, approx_amplitude))
            print(overlap_history[-1])

    print("Step Three")
    for i in range(2*N):
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5, lr_tau=0.5, lr_nstep=30, lr_min=1e-2)
        if compairing_exact_state is not None:
            compairing_exact_state = hadamard(compairing_exact_state, i)
            approx_amplitude = np.exp(op_model.model(all_state))
            overlap_history.append(exact_fidelity(
                compairing_exact_state, approx_amplitude))
            print(overlap_history[-1])
    if compairing_exact_state is not None:
        print("Overlap: ", overlap_history)
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


def bell_magic(state: RBM_flexable, n_chains: int, n_sweeps=None, n_samples=2**13, rndkey=1, compairing_exact_state=None):
    if n_sweeps is None:
        n_sweeps = 2 * state.in_features

    nv = basis_convert(state, compairing_exact_state)
    print("Finished basis convert")
    hi = nk.hilbert.Qubit(2*state.in_features)
    sampler_nv = nk.sampler.MetropolisLocal(
        hi, n_chains=n_chains, sweep_size=n_sweeps)
    sampler_nv_state = sampler_nv.init_state(nv, 1, jax.random.key(rndkey))
    samples_nv, sampler_nv_state = sampler_nv.sample(
        nv, 1, state=sampler_nv_state, chain_length=n_samples)
    samples_nv = samples_nv.reshape(4, -1, samples_nv.shape[-1])

    alpha = samples_nv[0] ^ samples_nv[1]
    beta = samples_nv[2] ^ samples_nv[3]

    ret = commutator(alpha, beta)
    # in total n_chains * n_samples / 4 samples
    return 2 * ret.sum() / (n_chains * n_samples / 4)
