import pytest
import rbm
import jax
import timeit
import netket as nk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def test_RBM_flexible_variational_state_interface():
    N = 6
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # init a random model(bias/local_bias/kernel all random number)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    kernel = np.array(model.kernel.value)

    # initialize the origin model
    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=20)
    original_parameters = vstate.parameters

    local_grad = rbm.log_wf_grad({"kernel": model.kernel.value, "bias": model.bias.value,
                                 "local_bias": model.local_bias.value}, all_state)
    avg_grad = jax.tree.map(lambda x: jnp.mean(x, axis=(0, 1)), local_grad)
    vstate.parameters = jax.tree.map(
        lambda x, y: x+y, vstate.parameters, avg_grad)

    assert original_parameters["bias"] + avg_grad["bias"] == pytest.approx(
        vstate.parameters["bias"])
    assert original_parameters["local_bias"] + avg_grad["local_bias"] == pytest.approx(
        vstate.parameters["local_bias"])
    assert original_parameters["kernel"] + avg_grad["kernel"] == pytest.approx(
        vstate.parameters["kernel"])
