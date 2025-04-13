import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import rbm
import pytest
import magic
import timeit
import mpi4py
import sys
from functools import reduce
from netket.utils import mpi

from netket.operator.spin import sigmax, sigmaz


def commutator_brute(alpha, beta):
    ret = np.zeros(alpha.shape[0])
    nspin = alpha.shape[1] // 2
    for i in range(alpha.shape[0]):
        for j in range(nspin):
            # ignore identity
            if (alpha[i, j] | alpha[i, j+nspin]) == 0:
                continue
            if (beta[i, j] | beta[i, j+nspin]) == 0:
                continue
            if alpha[i, j] == beta[i, j] and alpha[i, j+nspin] == beta[i, j+nspin]:
                continue
            ret[i] += 1
        ret[i] %= 2
    return ret


# note this is just for brute force magic calculation
def commutator_brute_magic(alpha, beta):
    ret = np.zeros(alpha.shape[0])
    nspin = alpha.shape[1] // 2
    for i in range(alpha.shape[0]):
        for j in range(0, 2*nspin, 2):
            # ignore identity
            if (alpha[i, j] | alpha[i, j+1]) == 0:
                continue
            if (beta[i, j] | beta[i, j+1]) == 0:
                continue
            if alpha[i, j] == beta[i, j] and alpha[i, j+1] == beta[i, j+1]:
                continue
            ret[i] += 1
        ret[i] %= 2
    return ret


def test_commutator():
    nbatch = 1000
    nspin = 34
    alpha = np.random.choice([0, 1], size=(nbatch, 2*nspin))
    beta = np.random.choice([0, 1], size=(nbatch, 2*nspin))

    commutator_magic = magic.commutator(alpha, beta)
    commutator_brute_force = commutator_brute(alpha, beta)
    assert np.array_equal(commutator_magic, commutator_brute_force)

    alpha = np.array([[0, 0, 0, 1, 1, 1],
                      [0, 1, 0, 1, 1, 0],
                      [0, 1, 1, 0, 1, 0]])
    beta = np.array([[0, 0, 1, 1, 0, 1],
                     [0, 1, 1, 0, 1, 0],
                     [0, 0, 0, 0, 1, 1]])
    # 01, 01, 01
    # 01, 11, 00 anticommute
    # 00, 11, 10
    # 00, 01, 01 commute
    commutator_brute_force = commutator_brute(alpha, beta)
    assert np.array_equal(commutator_brute_force, np.array([1, 0, 0]))

    # test runtime
    nbatch = 1000
    nspin = 12
    alpha = np.random.choice([0, 1], size=(nbatch, 2*nspin))
    beta = np.random.choice([0, 1], size=(nbatch, 2*nspin))

    f1 = timeit.timeit(lambda: commutator_brute(alpha, beta), number=1)
    f2 = timeit.timeit(lambda: magic.commutator(alpha, beta), number=1)
    print(f"brute force: {f1}, magic: {f2}")


def array2num(arr):
    return np.dot(arr, 2**np.arange(arr.size)[::-1])


def num2array(num, N):
    arr = np.zeros(N, dtype=int)
    for i in range(N):
        arr[i] = (num >> (N - i - 1)) & 1
    return arr


def compute(amplitude, basis, qubit, op):
    if op == 0:  # I
        return basis, 1
    elif op == 1:  # X
        return basis ^ (1 << qubit), 1
    elif op == 2:  # Z
        return basis, (-1)**((basis >> qubit) & 1)
    else:  # Y
        return basis ^ (1 << qubit), (-1)**(((basis >> qubit) & 1) ^ 1) * 1j


def test_compute():
    # qubit id: n-1, n-2, ..., 0
    amplitude = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j, 7+7j, 8+8j])
    # 000, 001, 010, 011, 100, 101, 110, 111
    assert compute(amplitude, 0, 0, 0) == (0, 1)  # 000 -> 000
    assert compute(amplitude, 0, 0, 1) == (1, 1)  # 000 -> 001
    assert compute(amplitude, 0, 0, 2) == (0, 1)  # 000 -> 000
    assert compute(amplitude, 0, 0, 3) == (1, -1j)  # 000 -> (-i) * 001

    assert compute(amplitude, 2, 2, 1) == (6, 1)  # 010 -> 110 X
    assert compute(amplitude, 4, 2, 2) == (4, (-1))  # 100 -> -100 Z
    assert compute(amplitude, 7, 1, 3) == (5, (1j))  # 111 -> (i) * 101 Y


def magic_brute(amplitude):
    nqubit = int(np.log2(amplitude.shape[0]))
    xi = np.zeros(4**nqubit, dtype=complex)
    for alpha in range(4**nqubit):
        for basis in range(2**nqubit):
            newbasis = basis
            coef = 1
            for qubit in range(nqubit):
                newbasis, newcoef = compute(amplitude, newbasis, qubit,
                                            (alpha // (4**qubit)) % 4)
                coef *= newcoef
            xi[alpha] += coef * amplitude[basis] * amplitude[newbasis].conj()
        assert xi[alpha] == pytest.approx(xi[alpha].conj())
        xi[alpha] = (xi[alpha]**2) / amplitude.shape[0]

    bell_magic = 0
    for alpha1 in range(4**nqubit):
        for alpha2 in range(4**nqubit):
            for beta1 in range(4**nqubit):
                for beta2 in range(4**nqubit):
                    a1 = alpha1 ^ alpha2
                    b1 = beta1 ^ beta2
                    commute = commutator_brute_magic(
                        np.expand_dims(num2array(a1, 2*nqubit), axis=0), np.expand_dims(num2array(b1, 2*nqubit), axis=0))
                    assert commute.shape == (1,)
                    if commute[0] == 0:
                        continue
                    # print(alpha1, alpha2, beta1, beta2, a1, b1)
                    bell_magic += xi[alpha1] * xi[alpha2] * \
                        xi[beta1] * xi[beta2] * 2
    return bell_magic.real


def additive_magic(theta, phi):
    ret = 0
    for n in range(theta.shape[0]):
        ret += np.log2(1 - 1/32 * (np.sin(theta[n])**2) * (35 + 28*np.cos(2*theta[n])
                       + np.cos(4*theta[n]) - 8*np.cos(4*phi[n])*(np.sin(theta[n])**4)))
    return -ret


def test_brute_magic():
    nqubit = 2
    theta = np.random.rand(nqubit) * np.pi * 4
    phi = np.random.rand(nqubit) * np.pi * 2

    states = np.concatenate(
        [np.expand_dims(np.cos(theta / 2), axis=1), np.expand_dims(np.sin(theta / 2) * np.exp(-1j * phi), axis=1)], axis=1)
    assert states.shape == (nqubit, 2)

    amplitude = reduce(np.kron, states)

    magic = magic_brute(amplitude)
    # print(magic)
    magic_additive = - np.log2(1 - magic)
    # print(magic_additive)

    # print(additive_magic(theta, phi))
    assert magic_additive == pytest.approx(additive_magic(theta, phi))


def test_rbm_magic():
    np.random.seed(10)
    nqubit = 15
    if mpi.rank == 0:
        theta = np.random.rand(nqubit) * np.pi * 4
        phi = np.random.rand(nqubit) * np.pi * 2
    else:
        theta = phi = None
    assert mpi.rank == mpi4py.MPI.COMM_WORLD.Get_rank()
    theta = mpi4py.MPI.COMM_WORLD.bcast(theta, root=0)
    phi = mpi4py.MPI.COMM_WORLD.bcast(phi, root=0)

    local_bias = -1j * phi
    bias = np.arccos(1 / 2 * np.cos(theta / 2)) * 1j
    kernel = np.diag((np.arccos(1 / 2 * np.sin(theta / 2)) -
                     np.arccos(1 / 2 * np.cos(theta / 2))) * 1j)

    state_rbm = rbm.RBM_flexable(nqubit, nqubit, rngs=jax.random.PRNGKey(0))
    state_rbm.reset(jnp.array(kernel), jnp.array(bias), jnp.array(local_bias))

    # make sure the rbm state is correct
    states = np.concatenate(
        [np.expand_dims(np.cos(theta / 2), axis=1), np.expand_dims(np.sin(theta / 2) * np.exp(-1j * phi), axis=1)], axis=1)
    assert states.shape == (nqubit, 2)

    amplitude = reduce(np.kron, states)
    hi = nk.hilbert.Qubit(nqubit)
    all_state = hi.all_states()
    amplitude_rbm = np.exp(state_rbm(all_state))

    assert amplitude_rbm == pytest.approx(amplitude)

    magic_rbm = magic.bell_magic(
        state_rbm, 8, n_samples=2**18, compairing_exact_state=None)
    magic_rbm_additive = - np.log2(1 - magic_rbm)
    print(magic_rbm_additive)
    print(additive_magic(theta, phi))
    print(magic_rbm)
    print(1 - np.exp2(-additive_magic(theta, phi)))


if __name__ == "__main__":
    test_rbm_magic()
