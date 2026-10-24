import numpy as np
import pytest

from fruitfly_brain.world import FlyGymWorld, OdysseyWorld


def test_odyssey_world_is_not_implemented():
    with pytest.raises(NotImplementedError):
        OdysseyWorld()


@pytest.fixture(scope="module")
def flygym_world():
    world = FlyGymWorld()
    yield world


def test_flygym_world_returns_eye_brightness_and_moves(flygym_world):
    eye = flygym_world.reset()
    assert np.asarray(eye).shape == (2,)
    start = np.asarray(flygym_world.position).copy()
    for _ in range(100):
        eye = flygym_world.step(np.array([1.0, 1.0]))
    assert np.asarray(eye).shape == (2,)
    assert np.all((np.asarray(eye) >= 0.0) & (np.asarray(eye) <= 1.0))
    moved = np.linalg.norm(np.asarray(flygym_world.position) - start)
    assert moved > 0.0
