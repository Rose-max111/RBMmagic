import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmaz
from tqdm import tqdm


class RBM_flexable(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: jax.Array):
        self.in_features = in_features
        self.out_features = out_features
        key1, key2 = jax.random.split(rngs)
        self.kernel = nnx.Param(0.01 * jax.random.normal(
            key1, (in_features, out_features)) + 1j * 0.01 * jax.random.normal(key2, (in_features, out_features)))
        self.bias = nnx.Param(jnp.zeros(out_features) +
                              1j * jnp.zeros(out_features))
        self.local_bias = nnx.Param(
            jnp.zeros(in_features) + 1j * jnp.zeros(in_features))

    def __call__(self, x: jax.Array):
        y = jnp.dot(x, self.kernel.value) + self.bias.value
        y = jnp.log(2 * jnp.cosh(y))
        return jnp.sum(y, axis=-1)+jnp.dot(x, self.local_bias.value)


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
