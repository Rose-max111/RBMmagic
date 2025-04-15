import pytest
import rbm
import jax
import timeit
import netket as nk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import timeit


def test_distribution():
    start = timeit.default_timer()
    N = 10
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # init a random model(bias/local_bias/kernel all random number)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))

    # initialize the origin model
    model.reset(jnp.zeros((N, N), dtype=complex), jnp.ones(
        N, dtype=complex) * 1j * jnp.pi / 3, jnp.array([np.log(2), np.log(2), 1/2 * np.pi*1j, 1/2 * np.pi*1j, 1/2 * np.pi*1j, np.log(2), np.log(2), 1/2 * np.pi*1j, 1/2 * np.pi*1j, 1/2 * np.pi*1j]))
    # initialize the model after a H gate (exact)

    # print(H_model.apply(model, all_state))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    # print(sampler)
    sampler_state = sampler.init_state(model, 1, jax.random.key(1))

    pdf_unnormalized = jnp.exp(model.apply(1, hi.all_states()))
    pdf_unnormalized = (pdf_unnormalized * pdf_unnormalized.conj()).real
    pdf = pdf_unnormalized / jnp.sum(pdf_unnormalized)

    def extract_pdf(row_samples, n_states):
        return jnp.sum((jnp.expand_dims(row_samples, axis=0) == jnp.expand_dims(
            n_states, axis=1)), axis=1)
    batched_extract_pdf = jax.vmap(extract_pdf, in_axes=(0, None))

    def estimate_pdf(n_samples):
        samples, _ = sampler.sample(
            model, 1, state=sampler_state, chain_length=n_samples)
        idxs = hi.states_to_numbers(samples)
        return jnp.sum(batched_extract_pdf(idxs, jnp.arange(hi.n_states)), axis=0) / (idxs.shape[0] * n_samples)

    estimated_pdf = estimate_pdf(2**13)
    # print("estimated_pdf = ", estimated_pdf)
    # print(pdf)
    print("error_sum = ", sum((np.abs(estimated_pdf - pdf)).real),
          "estimated_pdf = ", estimated_pdf, "pdf = ", pdf)
    end = timeit.default_timer()
    print("time = ", end - start)
    # The following is for plotting the pdf
    # plt.plot(pdf, label="exact")
    # plt.plot(estimate_pdf(2**10), label="2^10 samples")
    # plt.plot(estimate_pdf(2**14), '--', label="2^14 samples")

    # plt.ylim(0, 0.1)
    # plt.xlabel("hilbert space index")
    # plt.ylabel("pdf")
    # plt.legend()
    # plt.show()


def test_H_distribution():
    start = timeit.default_timer()
    N = 10
    hqubit = 1
    hi = nk.hilbert.Qubit(N)
    all_state = hi.all_states()
    # init a random model(bias/local_bias/kernel all random number)
    model = rbm.RBM_flexable(N, N, rngs=jax.random.PRNGKey(15))

    # initialize the origin model
    model.reset(jnp.zeros((N, N), dtype=complex), jnp.ones(
        N, dtype=complex) * 1j * jnp.pi / 3, jnp.array([np.log(2), np.log(2), 1/2 * np.pi*1j, 1/2 * np.pi*1j, 1/2 * np.pi*1j, np.log(2), np.log(2), 1/2 * np.pi*1j, 1/2 * np.pi*1j, 1/2 * np.pi*1j]))
    # initialize the model after a H gate (exact)
    Hmodel = rbm.RBM_H_State(model, hqubit)

    # print(H_model.apply(model, all_state))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    # print(sampler)
    sampler_state = sampler.init_state(Hmodel, 1, 2)

    pdf_unnormalized = jnp.exp(Hmodel.apply(1, hi.all_states()))
    pdf_unnormalized = (pdf_unnormalized * pdf_unnormalized.conj()).real
    pdf = pdf_unnormalized / jnp.sum(pdf_unnormalized)

    def extract_pdf(row_samples, n_states):
        return jnp.sum((jnp.expand_dims(row_samples, axis=0) == jnp.expand_dims(
            n_states, axis=1)), axis=1)
    batched_extract_pdf = jax.vmap(extract_pdf, in_axes=(0, None))

    def estimate_pdf(n_samples):
        samples, _ = sampler.sample(
            Hmodel, 1, state=sampler_state, chain_length=n_samples)
        idxs = hi.states_to_numbers(samples)
        return jnp.sum(batched_extract_pdf(idxs, jnp.arange(hi.n_states)), axis=0) / (idxs.shape[0] * n_samples)

    estimated_pdf = estimate_pdf(2**13)
    # print("estimated_pdf = ", estimated_pdf)
    # print(pdf)
    print("error_sum = ", sum((np.abs(estimated_pdf - pdf)).real),
          "estimated_pdf = ", estimated_pdf, "pdf = ", pdf)
    end = timeit.default_timer()
    print("time = ", end - start)
    # The following is for plotting the pdf
    # plt.plot(pdf, label="exact")
    # plt.plot(estimate_pdf(2**10), label="2^10 samples")
    # plt.plot(estimate_pdf(2**14), '--', label="2^14 samples")

    # plt.ylim(0, 0.1)
    # plt.xlabel("hilbert space index")
    # plt.ylabel("pdf")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    test_H_distribution()
