from unittest.mock import MagicMock

import numpy as np
import pytest
import imageio.v2 as imageio

from fruitfly_brain.world import FlyGymWorld, OdysseyWorld


def test_odyssey_world_is_not_implemented():
    with pytest.raises(NotImplementedError):
        OdysseyWorld()


@pytest.fixture(scope="module")
def flygym_world():
    world = FlyGymWorld()
    try:
        yield world
    finally:
        world.close()


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


def test_flygym_world_exports_decodable_mp4(flygym_world, tmp_path):
    flygym_world.reset()
    for _ in range(100):
        flygym_world.step(np.array([1.0, 1.0]))
        flygym_world.render()
    path = tmp_path / "nested" / "simulation.mp4"
    flygym_world.save_video(path)
    assert path.stat().st_size > 0
    with imageio.get_reader(path) as reader:
        assert reader.get_meta_data()["fps"] == 30
        frames = list(reader.iter_data())
    assert len(frames) >= 2
    assert all(frame.shape == (480, 640, 3) for frame in frames)
    assert np.any(frames[0] != frames[-1])


def test_flygym_world_rejects_empty_video_after_reset(flygym_world, tmp_path):
    flygym_world.reset()
    with pytest.raises(RuntimeError, match="No video frames recorded"):
        flygym_world.save_video(tmp_path / "empty.mp4")
    assert not (tmp_path / "empty.mp4").exists()


def test_gui_requires_desktop_display(monkeypatch, flygym_world):
    from fruitfly_brain import world as world_module

    monkeypatch.setattr(world_module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="requires a desktop display"):
        flygym_world.launch_viewer()


def test_gui_shares_existing_flygym_physics(monkeypatch, flygym_world):
    import mujoco
    import mujoco.viewer

    viewer = MagicMock()
    launch = MagicMock(return_value=viewer)
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    assert flygym_world.launch_viewer() is viewer
    physics = flygym_world._controller.physics
    launch.assert_called_once_with(physics.model.ptr, physics.data.ptr)
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.cam.trackbodyid == physics.model.name2id(
        flygym_world._controller.flies[0].thorax.full_identifier, "body",
    )
