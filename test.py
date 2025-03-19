import pytest
import rbm
import jax
import timeit
import netket as nk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

'''
def test_control_z():
    N = 5
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(0))
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    amplitude_all_state = np.exp(model(all_state))

    model.control_z(0, 1)
    model.control_z(1, 2)
    model.control_z(0, 4)
    amplitude_all_state_after = np.exp(model(all_state))

    for (state, i) in zip(all_state, range(2**N)):
        flag = (state[0] * state[1]) ^ (state[1]
                                        * state[2]) ^ (state[0] * state[4])
        if flag == 0:
                assert np.linalg.norm(
                    amplitude_all_state_after[i] - amplitude_all_state[i]) / np.linalg.norm(amplitude_all_state[i]) < 1e-5
                # print("amplitude_difference = ",
                #       amplitude_all_state_after[i] - amplitude_all_state[i], "amplitude = ", amplitude_all_state[i])
            else:
                assert np.linalg.norm(
                    amplitude_all_state_after[i] + amplitude_all_state[i]) / np.linalg.norm(amplitude_all_state[i]) < 1e-5
                # print("amplitude_difference = ",
                #       amplitude_all_state_after[i] + amplitude_all_state[i], "amplitude = ", amplitude_all_state[i])


def myrbm_call(params, x):
    kernel, bias, local_bias = params
    y = np.dot(x, kernel) + bias
    y = np.log(2 * np.cosh(y))
    return np.sum(y, axis=-1)+np.dot(x, local_bias)


def myrbm_grad(params, x):
    kernel, bias, local_bias = params
    local_bias_grad = x.copy()
    bias_grad = np.tanh(bias + np.dot(x, kernel))
    kernel_grad = np.expand_dims(x, axis=2) * np.expand_dims(bias_grad, axis=1)
    return (kernel_grad, bias_grad, local_bias_grad)


def test_rbm_log_wf_auto_grad():
    N = 10
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # init a random model(bias/local_bias/kernel all random number)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))
    bias = np.random.rand(N) + 1j * np.random.rand(N)
    local_bias = np.random.rand(N) + 1j * np.random.rand(N)
    kernel = np.array(model.kernel.value)

    model.reset(model.kernel.value, jnp.array(
        bias), jnp.array(local_bias))

    # test class params is the same as the numpy params
    assert model.kernel.value == pytest.approx(kernel)
    assert model.bias.value == pytest.approx(bias)
    assert model.local_bias.value == pytest.approx(local_bias)

    # test class call is the same as the numpy call
    model_log_wf = model(all_state)
    numpy_log_wf = myrbm_call((kernel, bias, local_bias), all_state)
    assert model_log_wf == pytest.approx(numpy_log_wf)
    # print("model_log_wf = ", model_log_wf, "numpy_log_wf = ", numpy_log_wf)

    # test class grad is the same as the numpy grad
    model_grad = rbm.batched_grad_fn(
        (model.kernel.value, model.bias.value, model.local_bias.value), all_state)
    numpy_grad = myrbm_grad((kernel, bias, local_bias), all_state)
    assert model_grad[2] == pytest.approx(numpy_grad[2])
    assert model_grad[1] == pytest.approx(numpy_grad[1])
    assert model_grad[0] == pytest.approx(numpy_grad[0])

    # evaluate the time of both method
    # time_model_grad = timeit.timeit(
    #     lambda: rbm.batched_grad_fn((model.kernel.value, model.bias.value, model.local_bias.value), all_state), number=100)
    # time_numpy_grad = timeit.timeit(
    #     lambda: myrbm_grad((kernel, bias, local_bias), all_state), number=100)
    # print("time_model_grad = ", time_model_grad,
    #       "time_numpy_grad = ", time_numpy_grad)


def test_rbm_H_state():
    N = 12
    H_qubit = 5
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
    # initialize the model after a H gate (exact)
    H_model = rbm.RBM_H_State(model, H_qubit)

    final_wf_direct = np.zeros(2**N, dtype=complex)
    for (state, i) in zip(all_state, range(2**N)):
        final_wf_direct[i] = np.exp(model((state))) * (1 - 2*state[H_qubit])
        another_state = state.at[H_qubit].set(1 - state[H_qubit])
        final_wf_direct[i] += np.exp(model((another_state)))
        final_wf_direct[i] /= np.sqrt(2)
        # print("state = ", state, "another_state = ", another_state,
        #   "final_wf_direct = ", final_wf_direct[i], "state[H_qubit] = ", state[H_qubit])
    final_log_wf_direct = np.log(final_wf_direct)
    final_log_wf_Hmodel = H_model(all_state)
    assert final_log_wf_direct == pytest.approx(final_log_wf_Hmodel)
'''


def test_rbm_H_state_distribution():
    N = 8
    H_qubit = 5
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
    # initialize the model after a H gate (exact)
    H_model = rbm.RBM_H_State(model, H_qubit)

    # print(H_model.apply(model, all_state))
    sampler = nk.sampler.MetropolisLocal(hi)
    sampler_state = sampler.init_state(H_model, 1, jax.random.key(1))

    pdf_unnormalized = jnp.exp(H_model.apply(1, hi.all_states()))
    pdf_unnormalized = (pdf_unnormalized * pdf_unnormalized.conj()).real
    pdf = pdf_unnormalized / jnp.sum(pdf_unnormalized)

    def extract_pdf(row_samples, n_states):
        return jnp.sum((jnp.expand_dims(row_samples, axis=0) == jnp.expand_dims(
            n_states, axis=1)), axis=1)
    batched_extract_pdf = jax.vmap(extract_pdf, in_axes=(0, None))

    def estimate_pdf(n_samples):
        samples, _ = sampler.sample(
            H_model, 1, state=sampler_state, chain_length=n_samples)
        idxs = hi.states_to_numbers(samples)
        return jnp.sum(batched_extract_pdf(idxs, jnp.arange(hi.n_states)), axis=0) / (idxs.shape[0] * n_samples)

    estimated_pdf = estimate_pdf(2**14)
    assert np.linalg.norm(estimated_pdf - pdf, ord=1) / \
        np.linalg.norm(pdf, ord=1) < 1e-1
    # print("relative_error = ", (np.abs(estimated_pdf - pdf) / pdf).real,
    #       "estimated_pdf = ", estimated_pdf, "pdf = ", pdf)
    # The following is for plotting the pdf
    # plt.plot(pdf, label="exact")
    # plt.plot(estimate_pdf(2**10), label="2^10 samples")
    # plt.plot(estimate_pdf(2**14), '--', label="2^14 samples")

    # plt.ylim(0, 0.1)
    # plt.xlabel("hilbert space index")
    # plt.ylabel("pdf")
    # plt.legend()
    # plt.show()
