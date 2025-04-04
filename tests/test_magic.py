import optimize
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import rbm
import pytest
import magic

from netket.operator.spin import sigmax, sigmaz


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


'''
    N = 4, seed = 18, PRNGkey=15, seems wrong, TB check
'''


def normalize(amplitude):
    return amplitude / np.linalg.norm(amplitude)


def test_hadamard_gate():
    N = 9
    np.random.seed(18)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    # initialize the origin model
    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))

    real_N = 2*N
    hi = nk.hilbert.Qubit(real_N)
    all_state = hi.all_states()  # the order of state is [N, N-1, ..., 0]
    real_model = rbm.state_preparation(model)

    overlap_history = []
    op_model = optimize.mcmc_optimize(
        real_model, nk.hilbert.Qubit(2*N), 4, 2*N, 2**13)
    exact_amplitude = np.exp(real_model(all_state))
    for i in range(N, 2*N):
        exact_amplitude = hadamard(exact_amplitude, i)
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5)
        approx_amplitude = np.exp(op_model.model(all_state))
        overlap_history.append(exact_fidelity(
            exact_amplitude, approx_amplitude))
        print(overlap_history[-1])

    for i in range(N):
        exact_amplitude = control_z(exact_amplitude, i, i+N)
        op_model.model.control_z(i, i+N)
        approx_amplitude = np.exp(op_model.model(all_state))
        overlap_history.append(exact_fidelity(
            exact_amplitude, approx_amplitude))
        print(overlap_history[-1])

    for i in range(2*N):
        exact_amplitude = hadamard(exact_amplitude, i)
        op_model.stochastic_reconfiguration_H(
            i, resample_phi=5, outprintconfig=5)
        approx_amplitude = np.exp(op_model.model(all_state))
        overlap_history.append(exact_fidelity(
            exact_amplitude, approx_amplitude))
        print(overlap_history[-1])
    for item in overlap_history:
        print(item, ",")

    # print(rbm_H_amplitude[0], exact_amplitude[0], approx_amplitude[0])
    # assert exact_amplitude == pytest.approx(approx_amplitude)
