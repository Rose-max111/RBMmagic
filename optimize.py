import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.linalg import solve
import timeit
import sys

from rbm import RBM_flexable, log_wf_grad, RBM_H_State


def exact_fidelity(psi, phi):
    return np.abs(np.sum(psi.conj()*phi)) / np.linalg.norm(psi) / np.linalg.norm(phi)


class mcmc_optimize():
    def __init__(self, model: RBM_flexable, hilbert, n_chains: int, n_sweeps: int, n_samples: int):
        self.model = model
        self.hilbert = hilbert
        self.n_chains = n_chains
        self.n_sweeps = n_sweeps
        self.n_samples = n_samples

    def fidelity(self, psi_phi, phi_phi, psi_psi, phi_psi):
        o1 = logsumexp(phi_psi - psi_psi, b=1.0 /
                       (phi_psi.shape[0]))
        o2 = logsumexp(psi_phi - phi_phi, b=1.0 /
                       (psi_phi.shape[0]))
        return np.exp(o1 + o2).real

    def fidelity_grad(self, phi_psi, psi_psi, O):
        term1 = jax.tree.map(lambda x: np.mean(x.conj(), axis=0), O)
        ratio = np.exp(phi_psi - psi_psi)
        term2_down = np.mean(ratio, axis=0)
        term2 = jax.tree.map(lambda x: np.mean(
            x.conj() * ratio.reshape((ratio.shape[0],)+(1,)*(x.ndim-1)), axis=0) / term2_down, O)
        return jax.tree.map(lambda x, y: x - y, term1, term2)

    def S_matrix(self, O):
        '''
        return the S matrix with order (kernel_flatted, bias, local_bias)
        '''
        merged_shaped_O = np.concatenate(
            (O["kernel"].reshape(O["kernel"].shape[0], -1), O["bias"], O["local_bias"]), axis=1)
        estO = merged_shaped_O - np.mean(merged_shaped_O, axis=0)
        return np.dot(estO.T.conj(), estO) / estO.shape[0]

    def sr_H_update(self, psi_phi, phi_phi, psi_psi, phi_psi, O, eps):
        F = self.fidelity(psi_phi, phi_phi, psi_psi, phi_psi)
        grad_F = self.fidelity_grad(phi_psi, psi_psi, O)

        S = self.S_matrix(O)
        S += eps * np.eye(S.shape[0])

        merged_grad_F = np.concatenate(
            (grad_F["kernel"].reshape(-1), grad_F["bias"], grad_F["local_bias"]), axis=0)

        merged_grad_params = solve(S, merged_grad_F, assume_a="her")
        delta_params = {"kernel": merged_grad_params[:grad_F["kernel"].size].reshape(
            grad_F["kernel"].shape), "bias": merged_grad_params[grad_F["kernel"].size:grad_F["kernel"].size+grad_F["bias"].size], "local_bias": merged_grad_params[grad_F["kernel"].size+grad_F["bias"].size:]}
        delta_params = jax.tree_map(lambda x: x*F, delta_params)

        return F, delta_params

    '''
    Noting!!
    We might should add a pertubation to the original model in case |psi_i> = 0 but \partial_i |psi_i> is not 0
    '''

    def stochastic_reconfiguration_H(self, qubit, tol=1e-3, lookback=5, max_iters=1000, resample_phi=None, lr=1e-1, lr_tau=None, lr_nstep=None, lr_min=0.0, eps=1e-4, rndkey_phi=0, rndkey_psi=1, outprintconfig=None):
        # init target model distribution and its sampler
        target_model = RBM_H_State(self.model, qubit)

        # add some pertubation to the model to avoid approximing zero
        pertubated_kernel_value = self.model.kernel.value + 0.01*np.random.normal(
            size=(self.model.in_features, self.model.out_features)) + 1j*0.01*np.random.normal(size=(self.model.in_features, self.model.out_features))
        pertubated_bias_value = self.model.bias.value + 0.01*np.random.normal(
            size=self.model.out_features) + 1j*0.01*np.random.normal(size=self.model.out_features)
        pertubated_local_bias_value = self.model.local_bias.value + 0.01*np.random.normal(
            size=self.model.in_features) + 1j*0.01*np.random.normal(size=self.model.in_features)
        self.model.reset(pertubated_kernel_value, pertubated_bias_value,
                         pertubated_local_bias_value)

        sampler_phi = nk.sampler.MetropolisLocal(
            self.hilbert, n_chains=self.n_chains, sweep_size=self.n_sweeps)
        sampler_phi_state = sampler_phi.init_state(
            target_model, 1, jax.random.key(rndkey_phi))
        # sample from the target model distribution
        samples_phi, sampler_phi_state = sampler_phi.sample(
            target_model, 1, state=sampler_phi_state, chain_length=self.n_samples // self.n_chains)
        # reshape samples to (n_chains * chain_length, nqubits)
        samples_phi = samples_phi.reshape(-1, samples_phi.shape[-1])
        # compute phi amplitude according phi distribution
        phi_phi = target_model(samples_phi)

        sampler_psi = nk.sampler.MetropolisLocal(
            self.hilbert, n_chains=self.n_chains, sweep_size=self.n_sweeps)
        vstate_psi = nk.vqs.MCState(
            sampler_psi, self.model, n_samples=self.n_samples)

        history = []
        F, F_mean_new, F_mean_old = 0.0, 0.0, 0.0
        diff_mean_F = 2*tol
        step = 0

        while (diff_mean_F > tol or step < 2*lookback+1) and F_mean_new < 1-tol and step < max_iters:
            start = timeit.default_timer()
            step += 1

            # sample from the model distribution
            samples_psi = vstate_psi.samples
            # reshape samples to (n_chains * chain_length, nqubits)
            samples_psi = samples_psi.reshape(-1, samples_psi.shape[-1])
            # compute psi amplitude according phi distribution
            psi_phi = self.model(samples_phi)
            # compute psi amplitude according psi distribution
            psi_psi = self.model(samples_psi)
            # compute phi amplitude according psi distribution
            phi_psi = target_model(samples_psi)

            O = log_wf_grad(vstate_psi.parameters, samples_psi)
            F, delta_params = self.sr_H_update(
                psi_phi, phi_phi, psi_psi, phi_psi, O, eps)

            if F < 0:
                print(self.model.kernel.value)
                print(self.model.bias.value)
                print(self.model.local_bias.value)
                sys.exit(1)

            vstate_psi.parameters = jax.tree.map(
                lambda x, y: x - lr * y, vstate_psi.parameters, delta_params)

            history.append(F)
            # print(f"step: {step}, F: {F}")
            # if outprintconfig is not None and step % outprintconfig == 1:
            # all_state = self.hilbert.all_states()
            # amplitude_psi = np.exp(self.model(all_state))
            # amplitude_phi = np.exp(target_model(all_state))
            # print(
            #     f"step: {step}, F: {F}, exact_fidelity: {exact_fidelity(amplitude_psi, amplitude_phi)**2}")
            self.model.reset(vstate_psi.parameters["kernel"], vstate_psi.parameters["bias"],
                             vstate_psi.parameters["local_bias"])

            if lr_tau is not None and step % lr_nstep == 0:
                lr = lr * lr_tau
                lr = max(lr, lr_min)

            if step > 2*lookback:
                F_mean_old = sum(history[-2*lookback:-lookback]) / lookback
                F_mean_new = sum(history[-lookback:]) / lookback

            diff_mean_F = np.abs(F_mean_new - F_mean_old)

            if resample_phi is not None and step % resample_phi == 0:
                samples_phi, sampler_phi_state = sampler_phi.sample(
                    target_model, 1, state=sampler_phi_state, chain_length=self.n_samples // self.n_chains)
                samples_phi = samples_phi.reshape(-1, samples_phi.shape[-1])
                phi_phi = target_model(samples_phi)

            end = timeit.default_timer()
            print(f"Time: {end - start}, Step: {step}, F: {F}")
        return history
