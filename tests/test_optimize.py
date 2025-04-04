import optimize
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import rbm
import pytest

from netket.operator.spin import sigmax, sigmaz


def exact_fidelity(psi, phi):
    return np.abs(np.sum(psi.conj()*phi)) / np.linalg.norm(psi) / np.linalg.norm(phi)


'''
def test_mcmc_fidelity():
    N = 10
    hqubit = 5
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # init a random model(bias/local_bias/kernel all random number)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    # initialize the origin model
    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))
    Hmodel = rbm.RBM_H_State(model, hqubit)

    opmodel = optimize.mcmc_optimize(model, hi, 4, 10, 1000)

    sampler_psi = nk.sampler.MetropolisLocal(hi, n_chains=4)
    vstate_psi = nk.vqs.MCState(sampler_psi, model, n_samples=2**13)

    sampler_phi = nk.sampler.MetropolisLocal(hi, n_chains=4)
    sampler_phi_state = sampler_phi.init_state(
        Hmodel, 1, jax.random.key(0))
    samples_phi, sampler_phi_state = sampler_phi.sample(
        Hmodel, 1, state=sampler_phi_state, chain_length=2**13)
    # according psi distribution
    samples_psi = vstate_psi.samples.reshape(-1, N)
    samples_phi = samples_phi.reshape(-1, N)

    psi_psi = model(samples_psi)
    phi_psi = Hmodel(samples_psi)
    psi_phi = model(samples_phi)
    phi_phi = Hmodel(samples_phi)
    fidelity = opmodel.fidelity(
        psi_phi, phi_phi, psi_psi, phi_psi)
    # print(fidelity)

    # Hoperator = 1/np.sqrt(2) * (sigmax(hi, hqubit)+sigmaz(hi, hqubit))
    # value = vstate_psi.expect(Hoperator)
    # # print(value)
    # print(value.mean * value.mean.conj())

    amplitude_psi = np.exp(model(all_state))
    amplitude_phi = np.exp(Hmodel(all_state))
    fidelity_exact = np.sum((amplitude_psi.conj()*amplitude_phi)) / \
        np.linalg.norm(amplitude_psi) / np.linalg.norm(amplitude_phi)
    # print(fidelity_exact**2)

    # assert np.abs((fidelity - fidelity_exact**2).real) <= 1e-2


def test_fidelity_grad():
    N = 8
    hi = nk.hilbert.Qubit(N)
    # define an arbitrary model, just for define the mcmc model
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    opmodel = optimize.mcmc_optimize(
        model, hi, 16, 10, 1000)  # define the mcmc model

    n_samples = 2
    phi_psi = np.array([[0.5-2j], [1.5+1j]])  # 2 samples
    psi_psi = np.array([[-0.5-1j], [2.0+3j]])  # 2 samples
    O = {"kernel": np.array([
        [[0.5-2j, 1.5+1j, 0.5+2j], [1.0, -1.0, -0.3]],
        [[-0.5-1j, 2.0+3j, 0.2+2j], [2.0, -2.0, 7-1j]]]), "bias":
        np.array([
            [0.5-2j, 1.5+1j],
            [-0.5-1j, 2.0+3j]]), "local_bias":
        np.array([[1.0, -1.0], [2.0, -2.0]])}
    assert O["kernel"].shape == (2, 2, 3)  # 2 samples, 6 parameters
    assert O["bias"].shape == (2, 2)  # 2 samples, 2 parameters
    assert O["local_bias"].shape == (2, 2)  # 2 samples, 2 parameters

    grad = opmodel.fidelity_grad(phi_psi, psi_psi, O)
    assert grad["kernel"].shape == (2, 3)
    assert grad["bias"].shape == (2,)
    assert grad["local_bias"].shape == (2,)

    term2_down = np.mean(np.exp(phi_psi - psi_psi))
    # print((
    #     1.5j) - ((0.5+2j)*np.exp((0.5-2j)-(-0.5-1j)) + (-0.5+1j)*np.exp((1.5+1j)-(2.0+3j))) / term2_down / 2
    # )
    # print(grad['kernel'][0, 0], term2_down)
    assert grad['kernel'][0, 0] == pytest.approx((
        1.5j) - ((0.5+2j)*np.exp((0.5-2j)-(-0.5-1j)) + (-0.5+1j)*np.exp((1.5+1j)-(2.0+3j))) / term2_down / 2)


def test_S_matrix():
    N = 8
    hi = nk.hilbert.Qubit(N)
    # define an arbitrary model, just for define the mcmc model
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    opmodel = optimize.mcmc_optimize(
        model, hi, 16, 10, 1000)  # define the mcmc model

    n_samples = 2
    O = {"kernel": np.array([
        [[0.5-2j, 1.5+1j, 0.5+2j], [1.0, -1.0+3j, -0.3]],
        [[-0.5-1j, 2.0+3j, 0.2+2j], [2.0, -2.0-1j, 7-1j]]]), "bias":
        np.array([
            [0.5-2j, 1.5+1j],
            [-0.5-1j, 2.0+3j]]), "local_bias":
        np.array([[1.0-2j, -1.0+2j], [2.0+3j, -2.0-1j]])}
    S = opmodel.S_matrix(O)
    # compute O[kernel:2,2 | local_bias:1]
    t1 = ((-1.0-3j)*(1.0-2j) + (-2.0+1j)*(2.0+3j))/2
    t2 = (-1.0-3j+-2.0+1j)/2
    t3 = (1.0-2j+2.0+3j)/2
    # print(t1-t2*t3)
    # print(S[4, 8])
    assert t1-t2*t3 == pytest.approx(S[4, 8])
'''


def test_hadamard_gate():
    N = 10
    hqubit = 5  # qubit id from 0 to N-1
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # define an arbitrary model, just for define the mcmc model
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    np.random.seed(51)
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    # initialize the origin model
    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))
    opmodel = optimize.mcmc_optimize(
        model, hi, 4, 8, 2**13)  # define the mcmc model

    exact_Hmodel = rbm.RBM_H_State(model, hqubit)
    fidelity_history = opmodel.stochastic_reconfiguration_H(
        hqubit, resample_phi=5, max_iters=1000, outprintconfig=5)

    exact_amplitude = np.exp(exact_Hmodel(all_state))
    approx_amplitude = np.exp(opmodel.model(all_state))
    fidelity = exact_fidelity(exact_amplitude, approx_amplitude)
    print(fidelity**2)

    # print(exact_amplitude.shape)
    # for i in range(256):
    #     print(exact_amplitude[i], approx_amplitude[i])
    assert fidelity_history[-1] == pytest.approx(fidelity**2, rel=1e-3)
