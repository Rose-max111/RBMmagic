import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import flax
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.linalg import solve
import timeit
import sys
from netket.utils import mpi
from mpi4py import MPI
import copy


from rbm import RBM_flexable, log_wf_grad, RBM_H_State
from jax import device_get


def exact_fidelity(psi, phi):
    return np.abs(np.sum(psi.conj()*phi)) / np.linalg.norm(psi) / np.linalg.norm(phi)


class mcmc_optimize():
    def __init__(self, model: RBM_flexable, hilbert, n_chains_per_rank: int, n_sweeps: int, n_samples: int):
        self.model = model
        self.hilbert = hilbert
        self.n_chains_per_rank = n_chains_per_rank
        self.n_sweeps = n_sweeps
        self.n_samples = n_samples

    def fidelity(self, psi_phi, phi_phi, psi_psi, phi_psi):
        o1 = logsumexp(phi_psi - psi_psi, b=1.0 /
                       (phi_psi.shape[0]))
        o2 = logsumexp(psi_phi - phi_phi, b=1.0 /
                       (psi_phi.shape[0]))
        # print(o1, o2, mpi.rank)
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
    We might should add a pertubation to the original model in case |psi_i> = 0 but partial_i |psi_i> is not 0
    '''

    def stochastic_reconfiguration_H(self, qubit, tol=1e-3, lookback=5, max_iters=1000, resample_phi=None, lr=1e-1, lr_tau=None, lr_nstep=None, lr_min=0.0, eps=1e-4, rndkey_phi=321, rndkey_psi=7723, outprintconfig=None):
        origin_model = copy.deepcopy(self.model)
        target_model = RBM_H_State(self.model, qubit)
        cnt = 0  # how many different pertubation are chosen
        while True:
            if origin_model.out_features > 3*origin_model.in_features:
                print(
                    "The hidden units are too large, please check the model")
                sys.exit(1)
            cnt += 1
            self.model = copy.deepcopy(origin_model)
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # add some pertubation to the model to avoid approximing zero
            if rank == 0:
                pertubated_kernel_value = self.model.kernel.value + 0.01*np.random.normal(
                    size=(self.model.in_features, self.model.out_features)) + 1j*0.01*np.random.normal(size=(self.model.in_features, self.model.out_features))
                pertubated_bias_value = self.model.bias.value + 0.01*np.random.normal(
                    size=self.model.out_features) + 1j*0.01*np.random.normal(size=self.model.out_features)
                pertubated_local_bias_value = self.model.local_bias.value + 0.01*np.random.normal(
                    size=self.model.in_features) + 1j*0.01*np.random.normal(size=self.model.in_features)
            else:
                pertubated_kernel_value = None
                pertubated_bias_value = None
                pertubated_local_bias_value = None
            pertubated_kernel_value = comm.bcast(
                pertubated_kernel_value, root=0)
            pertubated_bias_value = comm.bcast(pertubated_bias_value, root=0)
            pertubated_local_bias_value = comm.bcast(
                pertubated_local_bias_value, root=0)

            self.model.reset(jnp.array(pertubated_kernel_value), jnp.array(pertubated_bias_value),
                             jnp.array(pertubated_local_bias_value))

            sampler_phi = nk.sampler.MetropolisLocal(
                self.hilbert, n_chains_per_rank=self.n_chains_per_rank, sweep_size=self.n_sweeps)
            sampler_phi_state = sampler_phi.init_state(
                target_model, 1, rndkey_phi * (mpi.rank + 1))
            # sample from the target model distribution
            samples_phi, sampler_phi_state = sampler_phi.sample(
                # need discard some first samples
                target_model, 1, state=sampler_phi_state, chain_length=(self.n_samples // self.n_chains_per_rank // comm.Get_size()) + (self.n_samples) // 10)
            # reshape samples to (n_chains * chain_length, nqubits)
            samples_phi = samples_phi[:, (self.n_samples) //
                                      10:, :].reshape(-1, samples_phi.shape[-1]).block_until_ready()
            # compute phi amplitude according phi distribution
            phi_phi = target_model(samples_phi)
            phi_phi = comm.gather(phi_phi, root=0)
            if rank == 0:
                phi_phi = jnp.array(np.concatenate(phi_phi, axis=0))
                print(self.model.kernel.value.shape)
                print(self.model.bias.value.shape)

            sampler_psi = nk.sampler.MetropolisLocal(
                self.hilbert, n_chains_per_rank=self.n_chains_per_rank, sweep_size=self.n_sweeps)
            vstate_psi = nk.vqs.MCState(
                sampler_psi, self.model, n_samples=self.n_samples, seed=rndkey_psi * (mpi.rank + 1), sampler_seed=rndkey_psi * (mpi.rank + 1))

            history = []
            F, F_mean_new, F_mean_old = 0.0, 0.0, 0.0
            diff_mean_F = 2*tol
            step = 0

            while (diff_mean_F > tol or step < 2*lookback+1) and F_mean_new < 1-tol and step < max_iters:
                step += 1
                # sample from the model distribution
                start = timeit.default_timer()
                samples_psi = vstate_psi.samples
                # reshape samples to (n_chains * chain_length, nqubits)
                samples_psi = samples_psi.reshape(-1,
                                                  samples_psi.shape[-1]).block_until_ready()
                # compute psi amplitude according phi distribution
                psi_phi = self.model(samples_phi)
                # compute phi amplitude according psi distribution
                phi_psi = target_model(samples_psi)
                # compute psi amplitude according psi distribution
                psi_psi = self.model(samples_psi)

                phi_psi = comm.gather(phi_psi, root=0)
                psi_psi = comm.gather(psi_psi, root=0)
                psi_phi = comm.gather(psi_phi, root=0)
                samples_psi = comm.gather(samples_psi, root=0)

                if rank == 0:
                    phi_psi = jnp.array(np.concatenate(phi_psi, axis=0))
                    psi_psi = jnp.array(np.concatenate(psi_psi, axis=0))
                    psi_phi = jnp.array(np.concatenate(psi_phi, axis=0))
                    samples_psi = jnp.array(
                        np.concatenate(samples_psi, axis=0))

                if rank == 0:
                    O = log_wf_grad(vstate_psi.parameters, samples_psi)
                    F, delta_params = self.sr_H_update(
                        psi_phi, phi_phi, psi_psi, phi_psi, O, eps)
                else:
                    F = delta_params = None
                F = comm.bcast(F, root=0)
                delta_params = comm.bcast(delta_params, root=0)

                # After removing this line, the code became nearly a hundred times faster.
                vstate_psi.parameters = jax.tree.map(
                    lambda x, y: x - lr * y, vstate_psi.parameters, delta_params)

                self.model.reset(vstate_psi.parameters['kernel'], vstate_psi.parameters['bias'],
                                 vstate_psi.parameters['local_bias'])

                history.append(F)

                if lr_tau is not None and step % lr_nstep == 0:
                    lr = lr * lr_tau
                    lr = max(lr, lr_min)

                if step > 2*lookback:
                    F_mean_old = sum(history[-2*lookback:-lookback]) / lookback
                    F_mean_new = sum(history[-lookback:]) / lookback
                diff_mean_F = np.abs(F_mean_new - F_mean_old)

                if resample_phi is not None and step % resample_phi == 0:
                    samples_phi, sampler_phi_state = sampler_phi.sample(
                        target_model, 1, state=sampler_phi_state, chain_length=self.n_samples // self.n_chains_per_rank // comm.Get_size())
                    samples_phi = samples_phi.reshape(-1,
                                                      samples_phi.shape[-1])
                    phi_phi = target_model(samples_phi)
                    phi_phi = comm.gather(phi_phi, root=0)
                    if rank == 0:
                        phi_phi = jnp.array(np.concatenate(phi_phi, axis=0))
                end = timeit.default_timer()
                if rank == 0:
                    print(
                        f"step: {step}, F: {F}, time: {end - start}, mpi_rank: {mpi.rank}")
                # if outprintconfig is not None and step % outprintconfig == 1:
                # all_state = self.hilbert.all_states()
                # amplitude_psi = np.exp(self.model(all_state))
                # amplitude_phi = np.exp(target_model(all_state))
                # print(
                #     f"step: {step}, F: {F}, exact_fidelity: {exact_fidelity(amplitude_psi, amplitude_phi)**2}")
            if history[-1] > 0.95:
                return history
            else:
                if mpi.rank == 0:
                    print("Didn't find correct approximation:")
                    print("Restart the whole process(choose another pertubation)")
                if cnt == 2:  # three pertubation can not find correct approximation, increase hidden units
                    cnt = 0
                    # increase features
                    origin_model.expand_dims(
                        origin_model.in_features + origin_model.out_features)
                    if mpi.rank == 0:
                        print("Increase the hidden units")
