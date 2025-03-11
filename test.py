import pytest
import rbm
import jax
import netket as nk
import numpy as np


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
