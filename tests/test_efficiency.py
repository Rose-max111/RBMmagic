import pytest
import rbm
import jax
import timeit
import netket as nk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def test_efficiency():
    N = 30
    model = rbm.RBM_flexable(30, 30, rngs=jax.random.PRNGKey(0))
    np.random.seed(51)
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    # initialize the origin model
    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))
    input1 = jnp.array(np.random.choice([0, 1], size=(2**13, 30)))
    input2 = jnp.array(np.random.choice([0, 1], size=(2**13, 30)))

    target_model = rbm.RBM_H_State(model, 1)
    for i in range(5):
        start = timeit.default_timer()
        o1 = model(input1).block_until_ready()
        o2 = target_model(input2).block_until_ready()
        end = timeit.default_timer()
        new_bias = np.random.rand(N) + 1j * np.random.rand(N)
        new_local_bias = np.random.rand(N) + 1j * np.random.rand(N)
        new_kernel_value = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        # initialize the origin model
        model.reset(jnp.array(new_kernel_value), jnp.array(
            new_bias), jnp.array(new_local_bias))
        print("target time:", end - start)
