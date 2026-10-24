import numpy as np
import pandas as pd
import pytest

from fruitfly_brain.closed_loop import DATA_DIR
from fruitfly_brain.connectome import (
    build_connectome,
    load_saved_connectome,
    signed_synapse_weight,
)


def _edge(pre, post, syn, ach=1.0, gaba=0.0, glut=0.0, neuropil="ME_L"):
    return {
        "pre_pt_root_id": pre,
        "post_pt_root_id": post,
        "neuropil": neuropil,
        "syn_count": syn,
        "ach_avg": ach,
        "gaba_avg": gaba,
        "glut_avg": glut,
    }


def test_signed_weight_signs():
    edges = pd.DataFrame(
        [
            _edge(1, 2, 5, ach=1.0),
            _edge(1, 2, 5, ach=0.0, gaba=1.0),
            _edge(1, 2, 4, ach=0.0, glut=1.0),
        ]
    )
    np.testing.assert_allclose(signed_synapse_weight(edges), [5.0, -5.0, -4.0])


def test_duplicate_pairs_are_summed_and_unknown_ids_dropped():
    root_ids = np.array([10, 20, 30])
    edges = pd.DataFrame(
        [
            _edge(10, 20, 3),
            _edge(10, 20, 2),
            _edge(10, 999, 7),  # unknown post id: dropped
        ]
    )
    conn = build_connectome(edges, root_ids, min_pool_synapses=1000)
    assert conn.weights.shape == (3, 3)
    assert conn.weights[1, 0] == pytest.approx(5.0)
    assert conn.weights.nnz == 1


def test_weight_matrix_is_post_by_pre():
    root_ids = np.array([10, 20])
    edges = pd.DataFrame([_edge(10, 20, 4, gaba=1.0, ach=0.0)])
    conn = build_connectome(edges, root_ids, min_pool_synapses=1000)
    assert conn.weights[1, 0] == pytest.approx(-4.0)
    assert conn.weights[0, 1] == 0.0


def test_input_pool_requires_optic_synapses_and_fraction():
    root_ids = np.array([1, 2, 3, 9])
    rows = []
    # Neuron 1 gets 10 synapses in ME_L and nothing else -> in the left pool.
    rows.append(_edge(9, 1, 10, neuropil="ME_L"))
    # Neuron 2 gets 10 in ME_L but also 30 elsewhere -> 25% optic, excluded.
    rows.append(_edge(9, 2, 10, neuropil="ME_L"))
    rows.append(_edge(9, 2, 30, neuropil="AL_L"))
    # Neuron 3 gets 5 in LO_R -> below the synapse minimum, excluded.
    rows.append(_edge(9, 3, 5, neuropil="LO_R"))
    conn = build_connectome(pd.DataFrame(rows), root_ids)
    np.testing.assert_array_equal(conn.input_left, [0])
    np.testing.assert_array_equal(conn.input_right, [])


def test_motor_pool_uses_gng_output_and_side():
    root_ids = np.array([1, 2, 3, 4])
    rows = [
        # Neuron 0 sends 10 synapses to GNG, receives mostly on the left.
        _edge(1, 2, 10, neuropil="GNG"),
        _edge(4, 1, 20, neuropil="AL_L"),
        # Neuron 1 sends mostly outside GNG -> excluded.
        _edge(2, 3, 10, neuropil="GNG"),
        _edge(2, 4, 30, neuropil="AL_R"),
        # Neuron 2 sends 10 to GNG, receives mostly on the right.
        _edge(3, 4, 10, neuropil="GNG"),
        _edge(4, 3, 20, neuropil="AL_R"),
    ]
    conn = build_connectome(pd.DataFrame(rows), root_ids)
    np.testing.assert_array_equal(conn.motor_left, [0])
    np.testing.assert_array_equal(conn.motor_right, [2])


def test_saved_connectome_round_trip(tmp_path):
    root_ids = np.array([10, 20, 30])
    edges = pd.DataFrame([_edge(10, 20, 3, gaba=1.0, ach=0.0), _edge(30, 20, 2)])
    conn = build_connectome(edges, root_ids, min_pool_synapses=1000)
    from fruitfly_brain.connectome import save_connectome

    path = tmp_path / "c.npz"
    save_connectome(conn, path)
    loaded = load_saved_connectome(path)
    np.testing.assert_array_equal(loaded.root_ids, conn.root_ids)
    assert (loaded.weights != conn.weights).nnz == 0


@pytest.mark.skipif(
    not (DATA_DIR / "cache" / "connectome.npz").exists(),
    reason="cached FlyWire connectome not built",
)
def test_real_connectome_cache_is_sane():
    conn = load_saved_connectome(DATA_DIR / "cache" / "connectome.npz")
    assert conn.n_neurons == 139255
    assert conn.weights.nnz > 9_000_000
    assert conn.input_left.size > 0 and conn.input_right.size > 0
    assert conn.motor_left.size > 0 and conn.motor_right.size > 0
    assert np.intersect1d(conn.motor_left, conn.motor_right).size == 0
