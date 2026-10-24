"""Body-and-world environments the brain can control."""

from __future__ import annotations

import numpy as np

from flygym import Fly, YawOnlyCamera
from flygym.examples.locomotion import HybridTurningController

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
        camera = YawOnlyCamera(
            attachment_point=fly.model.worldbody,
            camera_name="camera_right",
            targeted_fly_names=fly.name,
        )
        self._controller = HybridTurningController(fly, cameras=[camera], timestep=1e-4)
        self.timestep = float(self._controller.timestep)
        self._obs: dict | None = None

    def reset(self) -> np.ndarray:
        self._obs, _ = self._controller.reset()
        return self.eye_brightness(self._obs)

    def step(self, drive: np.ndarray) -> np.ndarray:
        self._obs, *_ = self._controller.step(np.asarray(drive, dtype=float))
        return self.eye_brightness(self._obs)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self._obs["fly"][0], dtype=float)

    def render(self) -> None:
        self._controller.render()

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
