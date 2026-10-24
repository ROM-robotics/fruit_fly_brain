"""FlyWire proofread connectome loading and signed synapse-matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import sparse

EDGE_COLUMNS = (
    "pre_pt_root_id",
    "post_pt_root_id",
    "neuropil",
    "syn_count",
    "ach_avg",
    "gaba_avg",
    "glut_avg",
)

# Assumed neurotransmitter signs: ACh excitatory, GABA and glutamate inhibitory.
# Octopamine, serotonin and dopamine are not used.
VISUAL_NEUROPILS = {"L": ("ME_L", "LO_L"), "R": ("ME_R", "LO_R")}
OUTPUT_NEUROPIL = "GNG"


@dataclass(frozen=True)
class Connectome:
    """Signed connectome indexed by position in ``root_ids``.

    ``weights[post, pre]`` holds the signed synapse weight for each pair.
    The pools are arrays of neuron indices.
    """

    root_ids: np.ndarray
    weights: sparse.csr_matrix
    input_left: np.ndarray
    input_right: np.ndarray
    motor_left: np.ndarray
    motor_right: np.ndarray

    @property
    def n_neurons(self) -> int:
        return int(self.root_ids.shape[0])


def signed_synapse_weight(edges: pd.DataFrame) -> np.ndarray:
    """Synapse count scaled by (ACh - GABA - glutamate) transmitter probability."""
    signed = edges["ach_avg"] - edges["gaba_avg"] - edges["glut_avg"]
    return (edges["syn_count"] * signed).to_numpy(np.float32)


def _side_labels(neuropil: np.ndarray) -> np.ndarray:
    names = pd.Series(neuropil)
    ends_left = names.str.endswith("_L").to_numpy()
    ends_right = names.str.endswith("_R").to_numpy()
    return np.select([ends_left, ends_right], ["L", "R"], default="")


def _lookup(root_ids: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map root IDs to positions in ``root_ids``. Returns (positions, found_mask)."""
    order = np.argsort(root_ids)
    sorted_ids = root_ids[order]
    pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    found = sorted_ids[pos] == ids
    return order[pos], found


def build_connectome(
    edges: pd.DataFrame,
    root_ids: np.ndarray,
    *,
    min_pool_synapses: int = 10,
    min_pool_fraction: float = 0.5,
) -> Connectome:
    """Aggregate edges per neuron pair and select sensory and motor pools.

    Input pools: neurons with at least ``min_pool_synapses`` postsynaptic
    contacts in their eye's optic neuropils (ME/LO), and at least
    ``min_pool_fraction`` of their input there.

    Motor pools: neurons with at least ``min_pool_synapses`` presynaptic
    contacts in GNG and at least ``min_pool_fraction`` of their output there.
    Hemisphere comes from whichever side of the brain receives more input.
    """
    root_ids = np.asarray(root_ids)
    n = root_ids.shape[0]

    pre, pre_ok = _lookup(root_ids, edges["pre_pt_root_id"].to_numpy())
    post, post_ok = _lookup(root_ids, edges["post_pt_root_id"].to_numpy())
    keep = pre_ok & post_ok
    edges = edges[keep]
    pre = pre[keep]
    post = post[keep]

    neuropil = edges["neuropil"].astype(str).to_numpy()
    syn = edges["syn_count"].to_numpy(np.float64)
    pairs = pd.DataFrame({"post": post, "pre": pre, "w": signed_synapse_weight(edges)})
    agg = pairs.groupby(["post", "pre"], sort=False)["w"].sum()
    weights = sparse.csr_matrix(
        (
            agg.to_numpy(np.float32),
            (agg.index.get_level_values("post"), agg.index.get_level_values("pre")),
        ),
        shape=(n, n),
    )
    weights.eliminate_zeros()

    post_frame = pd.DataFrame(
        {"neuron": post, "neuropil": neuropil, "side": _side_labels(neuropil), "syn": syn}
    )
    pre_frame = pd.DataFrame({"neuron": pre, "neuropil": neuropil, "syn": syn})

    total_in = post_frame.groupby("neuron")["syn"].sum()
    input_pools = {}
    for side, optic in VISUAL_NEUROPILS.items():
        optic_in = post_frame[post_frame["neuropil"].isin(optic)].groupby("neuron")["syn"].sum()
        frac = optic_in / total_in.loc[optic_in.index]
        mask = ((optic_in >= min_pool_synapses) & (frac >= min_pool_fraction)).to_numpy()
        input_pools[side] = np.sort(optic_in.index[mask].to_numpy())

    lr = post_frame[post_frame["side"] != ""].groupby(["neuron", "side"])["syn"].sum()
    lr = lr.unstack(fill_value=0.0).reindex(columns=["L", "R"], fill_value=0.0)
    hemisphere = pd.Series(np.where(lr["L"] >= lr["R"], "L", "R"), index=lr.index)

    total_out = pre_frame.groupby("neuron")["syn"].sum()
    gng_out = pre_frame[pre_frame["neuropil"] == OUTPUT_NEUROPIL].groupby("neuron")["syn"].sum()
    frac = gng_out / total_out.loc[gng_out.index]
    mask = ((gng_out >= min_pool_synapses) & (frac >= min_pool_fraction)).to_numpy()
    candidates = gng_out.index[mask]
    side_of_candidate = hemisphere.reindex(candidates)
    motor_left = np.sort(candidates[(side_of_candidate == "L").to_numpy()].to_numpy())
    motor_right = np.sort(candidates[(side_of_candidate == "R").to_numpy()].to_numpy())

    return Connectome(
        root_ids=root_ids,
        weights=weights,
        input_left=input_pools["L"],
        input_right=input_pools["R"],
        motor_left=motor_left,
        motor_right=motor_right,
    )


def save_connectome(conn: Connectome, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        root_ids=conn.root_ids,
        data=conn.weights.data,
        indices=conn.weights.indices,
        indptr=conn.weights.indptr,
        shape=np.asarray(conn.weights.shape),
        input_left=conn.input_left,
        input_right=conn.input_right,
        motor_left=conn.motor_left,
        motor_right=conn.motor_right,
    )


def load_saved_connectome(path: str | Path) -> Connectome:
    with np.load(path) as f:
        weights = sparse.csr_matrix(
            (f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"])
        )
        return Connectome(
            root_ids=f["root_ids"],
            weights=weights,
            input_left=f["input_left"],
            input_right=f["input_right"],
            motor_left=f["motor_left"],
            motor_right=f["motor_right"],
        )


def load_connectome(
    edges_path: str | Path,
    root_ids_path: str | Path,
    cache_path: str | Path | None = None,
) -> Connectome:
    """Load from the cache if present; otherwise build from the FlyWire files and cache."""
    if cache_path is not None and Path(cache_path).exists():
        return load_saved_connectome(cache_path)
    edges = feather.read_table(edges_path, columns=list(EDGE_COLUMNS)).to_pandas()
    conn = build_connectome(edges, np.load(root_ids_path))
    if cache_path is not None:
        save_connectome(conn, cache_path)
    return conn
