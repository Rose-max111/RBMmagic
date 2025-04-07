import optimize
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import rbm
import pytest
import magic
import timeit

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


def test_commutator():
    nbatch = 10000
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
    nbatch = 5000
    nspin = 12
    alpha = np.random.choice([0, 1], size=(nbatch, 2*nspin))
    beta = np.random.choice([0, 1], size=(nbatch, 2*nspin))

    f1 = timeit.timeit(lambda: commutator_brute(alpha, beta), number=1)
    f2 = timeit.timeit(lambda: magic.commutator(alpha, beta), number=1)
    print(f"brute force: {f1}, magic: {f2}")
