"""Closed loop: FlyGym vision -> connectome SNN -> descending drive -> FlyGym body."""

from __future__ import annotations

import argparse
import os
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np

from fruitfly_brain.connectome import Connectome, load_connectome
from fruitfly_brain.lif import LIFConfig, LIFNetwork

if TYPE_CHECKING:
    from mujoco.viewer import Handle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAVE_VIDEO = False
DEFAULT_VIDEO_PATH = DATA_DIR / "runs" / "simulation.mp4"


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
    *,
    record_video: bool = False,
    viewer: Handle | None = None,
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
    completed_steps = 0
    next_viewer_sync = monotonic()

    for step in range(n_steps):
        if viewer is not None and not viewer.is_running():
            break
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
        completed_steps += 1
        if record_video:
            world.render()
        if viewer is not None:
            now = monotonic()
            if now >= next_viewer_sync:
                viewer.sync()
                next_viewer_sync = now + 1.0 / 60.0

    if viewer is not None and viewer.is_running():
        viewer.sync()

    return {
        "positions": positions[:completed_steps],
        "drives": np.asarray(drives).reshape(-1, 2),
        "motor_rates_hz": np.asarray(motor_rates).reshape(-1, 2),
        "spike_counts": np.asarray(spike_counts),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1.0, help="simulated seconds")
    parser.add_argument("--edges", type=Path, default=DATA_DIR / "proofread_connections_783.feather")
    parser.add_argument("--root-ids", type=Path, default=DATA_DIR / "proofread_root_ids_783.npy")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cache" / "connectome.npz")
    parser.add_argument("--out", type=Path, default=None, help="optional .npz for the run record")
    parser.add_argument("--gui", action="store_true", help="open the native MuJoCo desktop viewer")
    video_options = parser.add_mutually_exclusive_group()
    video_options.add_argument(
        "--video",
        type=Path,
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        metavar="MP4",
        help="record an MP4 (default path: data/runs/simulation.mp4)",
    )
    video_options.add_argument(
        "--no-video", dest="video", action="store_const", const=None,
        help="disable MP4 recording even when SAVE_VIDEO is True",
    )
    parser.set_defaults(video=DEFAULT_VIDEO_PATH if SAVE_VIDEO else None)
    args = parser.parse_args(argv)
    if args.video is not None and args.video.suffix.lower() != ".mp4":
        parser.error("--video must have an .mp4 extension")
    if args.gui and (
        os.environ.get("MUJOCO_GL", "glfw").lower() != "glfw"
        or os.environ.get("PYOPENGL_PLATFORM", "").lower() in {"egl", "osmesa"}
    ):
        parser.error(
            "--gui requires GLFW to avoid offscreen-context conflicts. Run with "
            "'env -u PYOPENGL_PLATFORM MUJOCO_GL=glfw python "
            "-m fruitfly_brain.closed_loop --gui'."
        )

    from fruitfly_brain.world import FlyGymWorld

    conn = load_connectome(args.edges, args.root_ids, args.cache)
    print(
        f"neurons={conn.n_neurons} synapse_pairs={conn.weights.nnz} "
        f"input L/R={len(conn.input_left)}/{len(conn.input_right)} "
        f"motor L/R={len(conn.motor_left)}/{len(conn.motor_right)}"
    )

    cfg = LoopConfig()
    with closing(FlyGymWorld()) as world:
        net = LIFNetwork(conn.weights, LIFConfig(dt_ms=cfg.brain_every * world.timestep * 1e3))
        viewer_context = world.launch_viewer() if args.gui else nullcontext()
        with viewer_context as viewer:
            if args.gui:
                print("MuJoCo GUI opened; close the window to stop early.", flush=True)
            record = run_closed_loop(
                conn, net, world, args.duration, cfg,
                record_video=args.video is not None, viewer=viewer,
            )

        if args.gui and len(record["positions"]) < int(round(args.duration / world.timestep)):
            print(f"GUI closed early after {len(record['positions'])} physics steps.")
        if len(record["positions"]) == 0:
            print("No simulation steps completed; no output saved.")
            return

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
        if args.video is not None:
            world.save_video(args.video)
            print(f"saved video {args.video}")


if __name__ == "__main__":
    main()
