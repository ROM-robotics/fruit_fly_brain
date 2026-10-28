# Fruit fly brain

FlyWire connectome-based spiking neural network connected to a FlyGym /
NeuroMechFly body in MuJoCo.

- [Project setup, data downloads and usage](README/01_README.md)
- [Code structure and architecture](README/02_CODE.md)

## MP4 preview (no GUI required)

The global `SAVE_VIDEO` flag in
[fruitfly_brain/closed_loop.py](fruitfly_brain/closed_loop.py) controls automatic
MP4 recording. It defaults to `False` to preserve numerical-only runs. Set it
to `True` to save `data/runs/simulation.mp4` on ordinary CLI runs.

You can also override the flag per run without editing code:

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python \
  -m fruitfly_brain.closed_loop \
  --duration 1.0 \
  --out data/runs/run_1s.npz \
  --video data/runs/run_1s.mp4
```

- `--video` alone uses the default MP4 path; `--no-video` disables recording
  even if `SAVE_VIDEO = True`.
- The existing side camera follows the fly at 640x480, 30 FPS and 0.2x playback
  speed: one simulated second produces approximately five seconds of video.
- Frames are captured during the simulation and encoded afterwards, including
  the initial stabilization period. Existing `.npz` files cannot be converted
  directly to video; re-run the simulation with recording enabled.
- Recording costs extra rendering time and buffers RGB frames in memory
  (roughly 132 MiB per simulated second). Start with short runs. Existing
  video files at the selected path are overwritten.
- An MP4 encoder is provided by FlyGym's ImageIO/FFmpeg dependencies. Encoding
  errors propagate rather than being reported as a successful export.

## Native MuJoCo GUI (host desktop)

Run the same closed-loop simulation in MuJoCo's desktop viewer:

```bash
env -u PYTHONPATH -u PYOPENGL_PLATFORM MUJOCO_GL=glfw .venv/bin/python \
  -m fruitfly_brain.closed_loop \
  --gui --duration 30.0 --no-video \
  --out data/runs/gui_run.npz
```

- A desktop display is required (Linux: `DISPLAY`, e.g. a local X11 desktop
  or XWayland). This is not a browser UI and does not work on a headless server
  without a display.
- Use `MUJOCO_GL=glfw`, not the headless `egl`/`osmesa` backends. Clear
  `PYOPENGL_PLATFORM` as shown to avoid conflicting OpenGL contexts. The CLI
  rejects those conflicting settings with an actionable error.
- The viewer follows the fly's thorax. Left-drag rotates, right-drag pans and
  the scroll wheel zooms. Camera selection is also available in MuJoCo's UI.
- Physics and SNN computation remain in our loop, not a second viewer-owned
  physics simulation. Viewer updates are limited to approximately 60 Hz
  wall-clock time; simulation speed depends on your machine, not a real-time
  guarantee. Native viewer pause/speed controls do not control this loop.
- The window closes when the requested simulated duration finishes. Close it
  earlier to stop; only completed steps are saved. Closing before the first
  step prints a message and skips output.
- Replace `--no-video` with `--video data/runs/gui_run.mp4` to record while
  viewing. Keep recorded runs short because video buffers frames in memory.

## Docker (Ubuntu 22.04, Python 3.10)

[docker/Dockerfile](docker/Dockerfile) installs dependencies directly into the
container's system Python with `python3 -m pip`; it does not create or copy a
virtual environment. Mesa OSMesa provides CPU-only headless rendering for the
fly's vision, without a display server or GPU passthrough. This can be slower
than GPU rendering. The same image also supports the native MuJoCo GUI through
local X11/XWayland display forwarding. GUI mode uses GLFW/GLX and Mesa software
rendering, so no GPU passthrough or NVIDIA toolkit is required.

### Prepare data

Before running a full simulation, download both FlyWire v783 files using the
instructions in the [setup guide, section 6](README/01_README.md):

```text
data/proofread_connections_783.feather
data/proofread_root_ids_783.npy
```

Both files are required unless a valid `data/cache/connectome.npz` already
exists. Follow the dataset's attribution and CC BY 4.0 license requirements
linked in the setup guide. Builds do not download data automatically.

### Build and run

From the project root:

```bash
bash docker/build_docker_image
bash docker/run_image --help
bash docker/run_image --duration 0.1 --out /data/runs/docker_smoke.npz
bash docker/run_image --duration 1.0 --video /data/runs/run_1s.mp4
```

The helpers also work from other working directories. Set `IMAGE_NAME` for a
custom image tag. The run helper mounts the project's `data/` at `/data`, so
downloaded data is available without rebuilding and caches/results persist on
the host. It runs with your host UID/GID to avoid root-owned output files.

Equivalent direct Docker commands:

```bash
docker build -f docker/Dockerfile -t romrobotics/fruit_fly_brain:latest .
docker run --rm \
  --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,source=$(pwd)/data,target=/data" \
  romrobotics/fruit_fly_brain:latest \
  --duration 0.1 --out /data/runs/docker_smoke.npz
```

The image also includes the local source data at `/data`; `/app/data` links
there so existing CLI defaults work unchanged. Without a bind mount, outputs
stay inside the container and are lost when it is removed with `--rm`.
Generated host caches/runs, local virtual environments and secrets are excluded
from the build context.

### Validate the image

```bash
docker run --rm --entrypoint python3 \
  romrobotics/fruit_fly_brain:latest -m pip check
docker run --rm --entrypoint python3 \
  romrobotics/fruit_fly_brain:latest -m pytest tests -q -p no:cacheprovider
```

The tests include a short headless FlyGym physics/vision simulation. The
real-connectome test is skipped unless its cache is available; to include it,
mount `data/` as above after building the cache with a simulation run.
The CLI explicitly releases simulation/rendering resources on completion or
failure, including OSMesa contexts before Python's thread-pool shutdown.

### Run the MuJoCo GUI in Docker

On a Linux desktop with X11 or XWayland, run from the project root:

```bash
# Ubuntu: install only if the host does not already have xauth.
sudo apt-get install xauth

bash docker/build_docker_image
bash docker/run_image \
  --gui --duration 30.0 --no-video \
  --out /data/runs/docker_gui.npz
```

Running the helper without arguments opens the GUI for 30 simulated seconds
with video recording disabled:

```bash
bash docker/run_image
# Or, when your working directory is docker/:
./run_image
```

Explicit arguments are passed through unchanged. For example,
`bash docker/run_image --duration 1.0 --no-video` remains a headless run.
This helper default does not change the image's direct `docker run` defaults.

The helper detects `--gui` (including its no-argument default) and automatically:

- passes your local `DISPLAY` and mounts the X11 sockets read-only;
- creates a temporary mode-600 Xauthority file containing only the selected
  display's cookie, mounts it read-only and removes it when Docker exits;
- overrides the image's headless defaults with `MUJOCO_GL=glfw` and
  `PYOPENGL_PLATFORM=glx`;
- enables Mesa software rendering with `LIBGL_ALWAYS_SOFTWARE=1`;
- runs as your host UID/GID and keeps results in the host's `data/runs/`.

No `xhost +`, `--privileged`, host networking or GPU device access is needed.
Only run images you trust: X11 access allows an application to interact with
your desktop. Use the GUI helper in the foreground so its temporary
authentication file can be cleaned up on exit.

For GUI plus video, keep the recording short:

```bash
bash docker/run_image \
  --gui --duration 1.0 \
  --out /data/runs/docker_gui_1s.npz \
  --video /data/runs/docker_gui_1s.mp4
```

The camera controls, early-close behavior and duration limits are the same as
the host GUI described above. Software rendering and the whole-brain simulation
can be slower than real time. There is no interactive GUI in headless mode;
pass explicit arguments without `--gui` to retain the original OSMesa
numerical/MP4 workflow.

#### Display troubleshooting

- Check `echo "$DISPLAY"` and `echo "$XAUTHORITY"` in a terminal opened on
  your desktop. The helper accepts local displays such as `:0`, `:1` and
  `unix:0`, not SSH-forwarded/TCP displays such as `localhost:10.0`.
- If `XAUTHORITY` is unset, the helper uses `$HOME/.Xauthority`. For a desktop
  that stores it elsewhere (for example GDM), set `XAUTHORITY` to that session's
  actual readable file before invoking the helper. It does not weaken X server
  access control to work around missing cookies.
- A Wayland session needs XWayland and a valid `DISPLAY`. Native Wayland-only
  sessions, remote Docker daemons and Docker Desktop/Windows/macOS display
  forwarding are not configured by this helper.
- Do not pass `MUJOCO_GL=egl` or `PYOPENGL_PLATFORM=osmesa` for the GUI. Use the
  helper so its GLFW/GLX overrides are applied together.