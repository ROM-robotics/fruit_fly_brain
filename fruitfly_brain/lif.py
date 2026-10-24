"""Leaky integrate-and-fire network with sparse, signed synapses (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class LIFConfig:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    v_rest_mv: float = -65.0
    v_thresh_mv: float = -50.0
    v_reset_mv: float = -65.0
    t_ref_ms: float = 2.0
    # mV that a median-sized synapse adds to its target on each presynaptic spike.
    synapse_mv: float = 1.0


class LIFNetwork:
    """Current-based LIF neurons. Each spike adds its synaptic weight to the targets.

    ``weights[post, pre]`` is scaled so the median nonzero |weight| equals
    ``config.synapse_mv``. Spikes from one step affect the next step.
    """

    def __init__(self, weights: sparse.csr_matrix, config: LIFConfig = LIFConfig()):
        self.config = config
        self.n = weights.shape[0]
        magnitudes = np.abs(weights.data)
        scale = config.synapse_mv / np.median(magnitudes) if magnitudes.size else 1.0
        self.weights = (weights * scale).astype(np.float32).tocsr()
        self.v = np.full(self.n, config.v_rest_mv, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self._ref_steps = int(round(config.t_ref_ms / config.dt_ms))
        self._leak = np.float32(config.dt_ms / config.tau_m_ms)

    def reset(self) -> None:
        self.v[:] = self.config.v_rest_mv
        self.refractory[:] = 0
        self.spikes[:] = False

    def step(self, external_mv: np.ndarray | None = None) -> np.ndarray:
        """Advance one time step. Returns a boolean array of neurons that spiked."""
        cfg = self.config
        synaptic = self.weights @ self.spikes.astype(np.float32)
        v = self.v + self._leak * (cfg.v_rest_mv - self.v) + synaptic
        if external_mv is not None:
            v = v + external_mv

        in_refractory = self.refractory > 0
        self.refractory[in_refractory] -= 1
        v[in_refractory] = cfg.v_reset_mv

        fired = (~in_refractory) & (v >= cfg.v_thresh_mv)
        v[fired] = cfg.v_reset_mv
        self.refractory[fired] = self._ref_steps

        self.v = v
        self.spikes = fired
        return fired
