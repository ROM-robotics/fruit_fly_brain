from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from scipy import sparse

from fruitfly_brain import closed_loop
from fruitfly_brain.connectome import Connectome
from fruitfly_brain.lif import LIFNetwork


@pytest.fixture
def conn():
    return Connectome(
        root_ids=np.array([1, 2]),
        weights=sparse.csr_matrix((2, 2), dtype=np.float32),
        input_left=np.array([0]),
        input_right=np.array([1]),
        motor_left=np.array([0]),
        motor_right=np.array([1]),
    )


@pytest.fixture
def world():
    return Mock(
        timestep=1e-4,
        position=np.zeros(3),
        reset=Mock(return_value=np.ones(2)),
        step=Mock(return_value=np.ones(2)),
    )


@pytest.fixture
def gui_environment(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)


def test_video_recording_preserves_numerical_results(conn, world):
    baseline = closed_loop.run_closed_loop(conn, LIFNetwork(conn.weights), world, 0.002)
    world.render.assert_not_called()
    recorded = closed_loop.run_closed_loop(
        conn, LIFNetwork(conn.weights), world, 0.002, record_video=True,
    )
    assert world.render.call_count == 20
    for name in baseline:
        np.testing.assert_array_equal(recorded[name], baseline[name])


@pytest.mark.parametrize(
    ("save_video", "options", "expected_path"),
    [
        (False, [], None),
        (True, [], closed_loop.DEFAULT_VIDEO_PATH),
        (False, ["--video"], closed_loop.DEFAULT_VIDEO_PATH),
        (False, ["--video", "custom.mp4"], "custom.mp4"),
        (True, ["--no-video"], None),
    ],
)
def test_cli_video_flag_and_overrides(monkeypatch, conn, world, save_video, options, expected_path):
    from pathlib import Path

    from fruitfly_brain import world as world_module

    monkeypatch.setattr(closed_loop, "SAVE_VIDEO", save_video)
    monkeypatch.setattr(closed_loop, "load_connectome", Mock(return_value=conn))
    monkeypatch.setattr(world_module, "FlyGymWorld", Mock(return_value=world))
    closed_loop.main(["--duration", "0.002", *options])
    if expected_path is None:
        world.render.assert_not_called()
        world.save_video.assert_not_called()
    else:
        assert world.render.call_count == 20
        world.save_video.assert_called_once_with(Path(expected_path))
    world.launch_viewer.assert_not_called()
    world.close.assert_called_once_with()


def test_cli_rejects_non_mp4_output(capsys):
    with pytest.raises(SystemExit) as exc:
        closed_loop.main(["--video", "invalid.avi"])
    assert exc.value.code == 2
    assert "--video must have an .mp4 extension" in capsys.readouterr().err


def test_gui_preserves_numerical_results(conn, world):
    baseline = closed_loop.run_closed_loop(conn, LIFNetwork(conn.weights), world, 0.002)
    viewer = Mock(is_running=Mock(return_value=True))
    displayed = closed_loop.run_closed_loop(
        conn, LIFNetwork(conn.weights), world, 0.002, viewer=viewer,
    )
    assert viewer.sync.called
    world.render.assert_not_called()
    for name in baseline:
        np.testing.assert_array_equal(displayed[name], baseline[name])


def test_gui_sync_is_throttled(monkeypatch, conn, world):
    monkeypatch.setattr(closed_loop, "monotonic", Mock(return_value=1.0))
    viewer = Mock(is_running=Mock(return_value=True))
    closed_loop.run_closed_loop(conn, LIFNetwork(conn.weights), world, 0.002, viewer=viewer)
    assert viewer.sync.call_count == 2  # Initial update and final update only.


@pytest.mark.parametrize("steps_before_close", [0, 7, 11])
def test_gui_close_returns_only_completed_steps(conn, world, steps_before_close):
    viewer = Mock(
        is_running=Mock(side_effect=[True] * steps_before_close + [False, False]),
    )
    record = closed_loop.run_closed_loop(
        conn, LIFNetwork(conn.weights), world, 0.002, viewer=viewer,
    )
    brain_steps = (steps_before_close + 9) // 10
    assert record["positions"].shape == (steps_before_close, 3)
    assert record["drives"].shape == (brain_steps, 2)
    assert record["motor_rates_hz"].shape == (brain_steps, 2)
    assert record["spike_counts"].shape == (brain_steps,)
    assert world.step.call_count == steps_before_close


@pytest.mark.parametrize("record_video", [False, True])
def test_cli_gui_lifecycle(monkeypatch, conn, world, record_video, gui_environment):
    from fruitfly_brain import world as world_module

    viewer = MagicMock()
    viewer.__enter__.return_value = viewer
    viewer.is_running.return_value = True
    world.launch_viewer.return_value = viewer
    monkeypatch.setattr(closed_loop, "SAVE_VIDEO", False)
    monkeypatch.setattr(closed_loop, "load_connectome", Mock(return_value=conn))
    monkeypatch.setattr(world_module, "FlyGymWorld", Mock(return_value=world))
    options = ["--video"] if record_video else []
    closed_loop.main(["--gui", "--duration", "0.002", *options])
    world.launch_viewer.assert_called_once_with()
    viewer.__exit__.assert_called_once()
    if record_video:
        world.save_video.assert_called_once_with(closed_loop.DEFAULT_VIDEO_PATH)
    else:
        world.save_video.assert_not_called()


def test_cli_gui_closed_before_first_step(monkeypatch, conn, world, capsys, gui_environment):
    from fruitfly_brain import world as world_module

    viewer = MagicMock()
    viewer.__enter__.return_value = viewer
    viewer.is_running.return_value = False
    world.launch_viewer.return_value = viewer
    monkeypatch.setattr(closed_loop, "load_connectome", Mock(return_value=conn))
    monkeypatch.setattr(world_module, "FlyGymWorld", Mock(return_value=world))
    closed_loop.main(["--gui", "--video"])
    assert "No simulation steps completed; no output saved." in capsys.readouterr().out
    world.save_video.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "platform"),
    [("egl", ""), ("osmesa", ""), ("glfw", "egl"), ("glfw", "osmesa")],
)
def test_cli_gui_rejects_conflicting_offscreen_backends(monkeypatch, capsys, backend, platform):
    monkeypatch.setenv("MUJOCO_GL", backend)
    monkeypatch.setenv("PYOPENGL_PLATFORM", platform)
    with pytest.raises(SystemExit) as exc:
        closed_loop.main(["--gui"])
    assert exc.value.code == 2
    assert "--gui requires GLFW" in capsys.readouterr().err


def test_cli_releases_world_on_simulation_failure(monkeypatch, conn, world):
    from fruitfly_brain import world as world_module

    monkeypatch.setattr(closed_loop, "load_connectome", Mock(return_value=conn))
    monkeypatch.setattr(world_module, "FlyGymWorld", Mock(return_value=world))
    monkeypatch.setattr(closed_loop, "run_closed_loop", Mock(side_effect=RuntimeError("test failure")))
    with pytest.raises(RuntimeError, match="test failure"):
        closed_loop.main(["--duration", "0.002", "--no-video"])
    world.close.assert_called_once_with()
