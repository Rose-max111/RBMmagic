import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import netket as nk
import matplotlib.pyplot as plt
import copy
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm
from scipy.linalg import block_diag
import copy


class RBM_flexable(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: jax.Array):
        self.in_features = in_features
        self.out_features = out_features
        key1, key2 = jax.random.split(rngs)
        self.kernel = nnx.Param(0.01 * jax.random.normal(
            key1, (in_features, out_features)) + 1j * 0.01 * jax.random.normal(key2, (in_features, out_features)))
        self.bias = nnx.Param(jnp.zeros(out_features, dtype=complex))
        self.local_bias = nnx.Param(
            jnp.zeros(in_features, dtype=complex))

    def reset(self, kernel_value: jax.Array, bias_value: jax.Array, local_bias_value: jax.Array):
        self.kernel = nnx.Param(kernel_value)
        self.bias = nnx.Param(bias_value)
        self.local_bias = nnx.Param(local_bias_value)
        self.in_features = kernel_value.shape[0]
        self.out_features = kernel_value.shape[1]

    def update(self, delta_params: dict, lr: float):
        self.kernel.value = self.kernel.value - lr * delta_params["kernel"]
        self.bias.value = self.bias.value - lr * delta_params["bias"]
        self.local_bias.value = self.local_bias.value - \
            lr * delta_params["local_bias"]

    def __call__(self, x: jax.Array):
        y = jnp.dot(x, self.kernel.value) + self.bias.value
        y = jnp.log(2 * jnp.cosh(y))
        return jnp.sum(y, axis=-1)+jnp.dot(x, self.local_bias.value)

    def apply(self, pars, x: jax.Array):
        return self.__call__(x)

    def control_z(self, ctrl_qubit: int, target_qubit: int):
        self.out_features = self.out_features + 1

        # new_hidden_weight = 1j * np.arccos(1/np.sqrt(3))
        new_hidden_weights_arr = jnp.zeros(
            (self.in_features, 1), dtype=complex)
        new_hidden_weights_arr = new_hidden_weights_arr.at[(
            ctrl_qubit, 0)].set(-1j * np.pi / 3)
        new_hidden_weights_arr = new_hidden_weights_arr.at[(
            target_qubit, 0)].set(1j * np.arctan(2/np.sqrt(3)))
        self.kernel = nnx.Param(jnp.concatenate(
            [self.kernel.value, new_hidden_weights_arr], axis=1))
        self.bias = nnx.Param(jnp.concatenate(
            [self.bias.value, jnp.array([1j * np.pi / 3])], axis=0))

        # delta_local_bias = 1/2 * np.log(3)
        new_local_bias = self.local_bias.value.at[ctrl_qubit].add(
            -np.log(2))
        new_local_bias = new_local_bias.at[target_qubit].add(
            1/2 * np.log(7/3) + 1j * np.pi)

        self.local_bias = nnx.Param(new_local_bias)

    def expand_dims(self, out_features: int):
        assert out_features > self.out_features
        append_kernel_value = jnp.zeros(
            (self.in_features, out_features - self.out_features), dtype=complex)
        append_bias_value = jnp.ones(
            out_features - self.out_features, dtype=complex) * 1j * jnp.pi / 3
        self.kernel = nnx.Param(jnp.concatenate(
            [self.kernel.value, append_kernel_value], axis=1))
        self.bias = nnx.Param(jnp.concatenate(
            [self.bias.value, append_bias_value], axis=0))
        self.out_features = out_features

    def conj(self):
        self.kernel = nnx.Param(self.kernel.value.conj())
        self.bias = nnx.Param(self.bias.value.conj())
        self.local_bias = nnx.Param(self.local_bias.value.conj())


class RBM_H_State(nnx.Module):
    def __init__(self, model: RBM_flexable, qubit: int):
        self.model = copy.deepcopy(model)
        self.qubit = qubit

    def __call__(self, x: jax.Array):
        another_x = x.at[:, self.qubit].set(1 - x[:, self.qubit])
        x_val, another_x_val = self.model(x), self.model(another_x)
        return jnp.log((jnp.exp(another_x_val) + jnp.exp(x_val) * (1 - 2*x[:, self.qubit])) / np.sqrt(2))

    def apply(self, pars, x: jax.Array):
        return self.__call__(x)


def state_preparation(state: RBM_flexable):
    state_conj = copy.deepcopy(state)
    state_conj.conj()

    new_kernel_value = block_diag(
        state.kernel.value, state_conj.kernel.value)
    new_bias_value = np.concatenate(
        (state.bias.value, state_conj.bias.value), axis=0)
    new_local_bias_value = np.concatenate(
        (state.local_bias.value, state_conj.local_bias.value), axis=0)
    state_conj.reset(jnp.array(new_kernel_value), jnp.array(
        new_bias_value), jnp.array(new_local_bias_value))
    return state_conj


def RBM_call_params(params, x):
    '''
    params[0, 1, 2] correspond to kernel.value, bias.value and local_bias.value
    '''
    y = jnp.dot(x, params["kernel"]) + params["bias"]
    y = jnp.log(2 * jnp.cosh(y))
    return jnp.sum(y, axis=-1)+jnp.dot(x, params["local_bias"])


grad_fn = jax.grad(RBM_call_params, argnums=0, holomorphic=True)
batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0))


def log_wf_grad(params, x: jax.Array):
    '''
    x: shape = (n_chain * batch_size, N)
    '''
    return batched_grad_fn(params, x)


if __name__ == "__main__":
    N = 20
    Hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    Gamma = -1
    H = sum([Gamma * sigmax(Hilbert, i) for i in range(N)])
    V = -1
    H += sum([V * sigmaz(Hilbert, i) * sigmaz(Hilbert, (i + 1) % N)
              for i in range(N)])

    model = RBM_flexable(N, N, rngs=jax.random.PRNGKey(0))

    # model = nk.models.RBM(alpha=1)
    sampler = nk.sampler.MetropolisLocal(Hilbert)
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

    optimizer = nk.optimizer.Sgd(learning_rate=0.1)

    gs = nk.driver.VMC(
        H,
        optimizer,
        variational_state=vstate,
        preconditioner=nk.optimizer.SR(diag_shift=0.1),
    )

    log = nk.logging.RuntimeLog()
    gs.run(n_iter=300, out=log)

    ffn_energy = vstate.expect(H)
    # error = abs((ffn_energy.mean - eig_vals[0]) / eig_vals[0])
    print("Optimized energy and relative error: ", ffn_energy)

    # print(model.kernel.value)
