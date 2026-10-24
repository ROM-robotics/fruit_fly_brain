import numpy as np
from scipy import sparse

from fruitfly_brain.lif import LIFConfig, LIFNetwork


def _net(dense, synapse_mv=25.0):
    return LIFNetwork(sparse.csr_matrix(np.asarray(dense, dtype=np.float32)), LIFConfig(synapse_mv=synapse_mv))


def test_no_input_no_spikes():
    net = _net(np.zeros((3, 3)))
    total = sum(int(net.step().sum()) for _ in range(200))
    assert total == 0
    np.testing.assert_allclose(net.v, -65.0, atol=1e-4)


def test_refractory_limits_firing_rate():
    net = _net(np.zeros((1, 1)))
    spikes = [bool(net.step(np.array([100.0], dtype=np.float32))[0]) for _ in range(100)]
    count = sum(spikes)
    # Refractory period of 2 ms means at most one spike every 3 steps.
    assert 1 <= count <= 34


def test_excitatory_chain_propagates():
    # Neuron 0 -> neuron 1 with a 25 mV synapse.
    net = _net([[0, 0], [1, 0]])
    first = net.step(np.array([20.0, 0.0], dtype=np.float32))
    assert first[0] and not first[1]
    second = net.step()
    assert second[1]


def test_inhibitory_synapse_suppresses_firing():
    # Neuron 0 inhibits neuron 1. Neuron 1 is held just below threshold.
    net = _net([[0, 0], [-1, 0]])
    drive = np.array([20.0, 12.0], dtype=np.float32)
    net.step(drive)  # neuron 0 fires, neuron 1 settles near -53 mV
    after = net.step(np.array([0.0, 12.0], dtype=np.float32))
    assert not after[1]
    assert net.v[1] < -60.0


def test_reset_restores_resting_state():
    net = _net(np.zeros((2, 2)))
    net.step(np.array([20.0, 20.0], dtype=np.float32))
    net.reset()
    np.testing.assert_allclose(net.v, -65.0)
    assert not net.spikes.any()
    assert (net.refractory == 0).all()
