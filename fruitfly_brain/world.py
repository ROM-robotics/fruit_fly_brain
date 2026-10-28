"""Body-and-world environments the brain can control."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from flygym import Fly, YawOnlyCamera
from flygym.examples.locomotion import HybridTurningController

if TYPE_CHECKING:
    from mujoco.viewer import Handle

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
LEG_SEGMENTS = ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")


class FlyGymWorld:
    """MuJoCo fly body and arena from FlyGym, driven by its turning controller.

    ``step`` takes a descending drive ``[left, right]`` and returns the
    brightness of each eye, averaged over ommatidia and photoreceptor types.
    """

    def __init__(self) -> None:
        contact_sensors = [f"{leg}{seg}" for leg in LEGS for seg in LEG_SEGMENTS]
        fly = Fly(
            enable_adhesion=True,
            enable_vision=True,
            contact_sensor_placements=contact_sensors,
        )
        self._camera = YawOnlyCamera(
            attachment_point=fly.model.worldbody,
            camera_name="camera_right",
            targeted_fly_names=fly.name,
        )
        self._controller = HybridTurningController(fly, cameras=[self._camera], timestep=1e-4)
        self.timestep = float(self._controller.timestep)
        self._obs: dict | None = None
        self._has_video_frames = False

    def reset(self) -> np.ndarray:
        self._obs, _ = self._controller.reset()
        self._has_video_frames = False
        return self.eye_brightness(self._obs)

    def step(self, drive: np.ndarray) -> np.ndarray:
        self._obs, *_ = self._controller.step(np.asarray(drive, dtype=float))
        return self.eye_brightness(self._obs)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self._obs["fly"][0], dtype=float)

    def render(self) -> None:
        frames = self._controller.render()
        if frames[0] is not None:
            self._has_video_frames = True

    def save_video(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("Video output path must have an .mp4 extension")
        if not self._has_video_frames:
            raise RuntimeError("No video frames recorded; enable recording and run the simulation first")
        self._camera.save_video(path, stabilization_time=0)

    def launch_viewer(self) -> Handle:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "--gui requires a desktop display (DISPLAY). Run on your desktop, "
                "or omit --gui and use --video for headless rendering."
            )
        import mujoco
        from mujoco.viewer import launch_passive

        physics = self._controller.physics
        viewer = launch_passive(physics.model.ptr, physics.data.ptr)
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = physics.model.name2id(
                self._controller.flies[0].thorax.full_identifier, "body",
            )
            viewer.cam.distance = 10.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0
        return viewer

    def close(self) -> None:
        try:
            self._controller.close()
        finally:
            self._controller.physics.free()

    @staticmethod
    def eye_brightness(obs: dict) -> np.ndarray:
        vision = np.asarray(obs["vision"], dtype=float)
        if vision.max(initial=0.0) > 1.0:
            vision = vision / 255.0
        return vision.mean(axis=(1, 2))


class OdysseyWorld:
    """Placeholder for an Odyssey-3 backend.

    Odyssey has not published a public API for Odyssey-3, so no working
    backend exists yet. Construction fails explicitly to avoid silent fallback.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Odyssey-3 has no public API. Use FlyGymWorld, or implement an adapter "
            "once Odyssey publishes an interface."
        )
