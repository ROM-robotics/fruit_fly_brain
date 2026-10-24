"""Closed loop: FlyGym vision -> connectome SNN -> descending drive -> FlyGym body."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fruitfly_brain.connectome import Connectome, load_connectome
from fruitfly_brain.lif import LIFConfig, LIFNetwork

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class LoopConfig:
    # Physics steps per SNN update. With the 0.1 ms physics step and 1 ms LIF step, this is 10.
    brain_every: int = 10
    # mV added per SNN step to visual input neurons, per unit eye brightness.
    input_gain_mv: float = 3.0
    base_drive: float = 0.8
    motor_gain: float = 0.6
    # Motor pool rate (Hz) that maps to full motor gain.
    max_motor_rate_hz: float = 100.0


def _pool_rate_hz(spikes: np.ndarray, pool: np.ndarray, dt_s: float) -> float:
    if pool.size == 0:
        return 0.0
    return float(spikes[pool].mean() / dt_s)


def run_closed_loop(
    conn: Connectome,
    net: LIFNetwork,
    world,
    duration_s: float,
    cfg: LoopConfig = LoopConfig(),
) -> dict:
    """Run the loop and return trajectory and activity records."""
    brain_dt_s = net.config.dt_ms * 1e-3
    if not np.isclose(brain_dt_s, cfg.brain_every * world.timestep):
        raise ValueError(
            f"LIF dt ({brain_dt_s} s) must equal brain_every * physics step "
            f"({cfg.brain_every * world.timestep} s)"
        )

    n_steps = int(round(duration_s / world.timestep))
    external = np.zeros(conn.n_neurons, dtype=np.float32)
    eye = world.reset()
    drive = np.full(2, cfg.base_drive)

    positions = np.zeros((n_steps, 3))
    drives: list[np.ndarray] = []
    motor_rates: list[np.ndarray] = []
    spike_counts: list[int] = []

    for step in range(n_steps):
        if step % cfg.brain_every == 0:
            external[:] = 0.0
            external[conn.input_left] = cfg.input_gain_mv * eye[0]
            external[conn.input_right] = cfg.input_gain_mv * eye[1]
            spikes = net.step(external)

            rates = np.array(
                [
                    _pool_rate_hz(spikes, conn.motor_left, brain_dt_s),
                    _pool_rate_hz(spikes, conn.motor_right, brain_dt_s),
                ]
            )
            normalized = np.clip(rates / cfg.max_motor_rate_hz, 0.0, 1.0)
            drive = cfg.base_drive + cfg.motor_gain * normalized
            drives.append(drive.copy())
            motor_rates.append(rates)
            spike_counts.append(int(spikes.sum()))

        eye = world.step(drive)
        positions[step] = world.position

    return {
        "positions": positions,
        "drives": np.asarray(drives),
        "motor_rates_hz": np.asarray(motor_rates),
        "spike_counts": np.asarray(spike_counts),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1.0, help="simulated seconds")
    parser.add_argument("--edges", type=Path, default=DATA_DIR / "proofread_connections_783.feather")
    parser.add_argument("--root-ids", type=Path, default=DATA_DIR / "proofread_root_ids_783.npy")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cache" / "connectome.npz")
    parser.add_argument("--out", type=Path, default=None, help="optional .npz for the run record")
    args = parser.parse_args(argv)

    from fruitfly_brain.world import FlyGymWorld

    conn = load_connectome(args.edges, args.root_ids, args.cache)
    print(
        f"neurons={conn.n_neurons} synapse_pairs={conn.weights.nnz} "
        f"input L/R={len(conn.input_left)}/{len(conn.input_right)} "
        f"motor L/R={len(conn.motor_left)}/{len(conn.motor_right)}"
    )

    cfg = LoopConfig()
    world = FlyGymWorld()
    net = LIFNetwork(conn.weights, LIFConfig(dt_ms=cfg.brain_every * world.timestep * 1e3))
    record = run_closed_loop(conn, net, world, args.duration, cfg)

    displacement = np.linalg.norm(record["positions"][-1] - record["positions"][0])
    print(
        f"steps={len(record['positions'])} total_spikes={int(record['spike_counts'].sum())} "
        f"mean_drive={record['drives'].mean(axis=0).round(3).tolist()} "
        f"displacement_mm={displacement:.3f}"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **record)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
