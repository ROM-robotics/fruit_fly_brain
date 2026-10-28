# Fruit fly brain

FlyWire connectome ဒေတာကို အခြေခံထားသော spiking neural network ကို MuJoCo ထဲရှိ
FlyGym / NeuroMechFly ကောင်ခန္ဓာနှင့် ချိတ်ဆက်ထားသည်။

- [Project setup, data downloads and usage](README/01_README.md)
- [Code structure and architecture](README/02_CODE.md)
- [Q&A (မေးခွန်းနှင့် အဖြေများ)](README/03_QA.md)

## MP4 ဗီဒီယို preview (GUI မလိုအပ်ပါ)

[fruitfly_brain/closed_loop.py](fruitfly_brain/closed_loop.py) ထဲရှိ global
`SAVE_VIDEO` flag က MP4 ဗီဒီယို အလိုအလျောက် သိမ်းခြင်းကို ထိန်းချုပ်သည်။
ပုံမှန်အားဖြင့် `False` ဖြစ်ပြီး ဂဏန်းဒေတာသာ ထွက်သော run များကို ထိန်းသိမ်းထားသည်။
ပုံမှန် CLI run များတွင် `data/runs/simulation.mp4` ကို သိမ်းလိုပါက `True` သို့
ပြောင်းပါ။

Code ကို မပြင်ဘဲ run တစ်ခုချင်းအတွက်လည်း flag ကို override လုပ်နိုင်သည်:

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python \
  -m fruitfly_brain.closed_loop \
  --duration 1.0 \
  --out data/runs/run_1s.npz \
  --video data/runs/run_1s.mp4
```

- `--video` တစ်ခုတည်းသုံးလျှင် ပုံမှန် MP4 လမ်းကြောင်းကို သုံးမည်; `SAVE_VIDEO = True`
  ဖြစ်နေလျှင်ပင် `--no-video` က ဗီဒီယို သိမ်းခြင်းကို ပိတ်ထားမည်။
- ရှိနေပြီးသား ဘေးကင်မရာသည် 640x480၊ 30 FPS၊ 0.2x ပြန်ဖွင့်မြန်နှုန်းဖြင့် fly ကို
  နောက်လိုက်သည်: simulation ၁ စက္ကန့်ကို ဗီဒီယို ခန့်မှန်းခြေ ၅ စက္ကန့်ထုတ်ပေးသည်။
- Frame များကို simulation အစ stabilization ကာလအပါအဝင် ဖမ်းယူပြီး နောက်မှ encode
  ပြုလုပ်သည်။ ရှိနှင့်ပြီးသား `.npz` ဖိုင်များကို ဗီဒီယိုအဖြစ် တိုက်ရိုက်ပြောင်း၍ မရပါ;
  recording ဖွင့်ထားပြီး simulation ကို ပြန်လည် run ရမည်။
- Recording က rendering အချိန်ပိုကုန်ပြီး RGB frame များကို memory ထဲ buffer
  လုပ်ထားသည် (simulation ၁ စက္ကန့်လျှင် ခန့်မှန်းခြေ 132 MiB)။ Run တိုတိုများဖြင့်
  စတင်သင့်သည်။ ရွေးချယ်ထားသော လမ်းကြောင်းရှိ ဗီဒီယိုဖိုင်များကို ထပ်ရေးအစားထိုးမည်။
- MP4 encoder ကို FlyGym ၏ ImageIO/FFmpeg dependency များက ပံ့ပိုးသည်။ Encoding
  အမှားများသည် အောင်မြင်သော export အဖြစ် သတင်းပို့မည့်အစား ပျံ့နှံ့ (propagate)
  သွားမည်။

## Native MuJoCo GUI (host desktop)

Closed-loop simulation အတူတူကို MuJoCo ၏ desktop viewer တွင် run ရန်:

```bash
env -u PYTHONPATH -u PYOPENGL_PLATFORM MUJOCO_GL=glfw .venv/bin/python \
  -m fruitfly_brain.closed_loop \
  --gui --duration 30.0 --no-video \
  --out data/runs/gui_run.npz
```

- Desktop display လိုအပ်သည် (Linux: `DISPLAY`၊ ဥပမာ local X11 desktop သို့မဟုတ်
  XWayland)။ ၎င်းသည် browser UI မဟုတ်ပဲ display မရှိသော headless server တွင်
  အလုပ်မလုပ်ပါ။
- Headless `egl`/`osmesa` backend များမဟုတ်ဘဲ `MUJOCO_GL=glfw` ကို သုံးပါ။
  OpenGL context များ ပဋိပက္ခမဖြစ်စေရန် ပြထားသည့်အတိုင်း `PYOPENGL_PLATFORM`
  ကို clear လုပ်ပါ။ ဤပဋိပက္ခဖြစ်စေသော setting များကို CLI က လုပ်ဆောင်နိုင်သည့်
  error ဖြင့် ငြင်းပယ်မည်။
- Viewer သည် fly ၏ thorax ကို နောက်လိုက်သည်။ ဘယ်ဘက်ဆွဲလျှင် rotate၊ ညာဘက်ဆွဲလျှင်
  pan၊ scroll wheel ဖြင့် zoom လုပ်နိုင်သည်။ MuJoCo UI တွင် camera ရွေးချယ်ခွင့်
  လည်း ရရှိနိုင်သည်။
- Physics နှင့် SNN တွက်ချက်မှုသည် ဒုတိယ viewer ပိုင် physics simulation တစ်ခု
  မဟုတ်ဘဲ ကျွန်ုပ်တို့၏ loop ထဲမှာပင် ကျန်ရှိနေသည်။ Viewer update များသည်
  wall-clock အချိန် ခန့်မှန်းခြေ 60 Hz ဖြင့် ကန့်သတ်ထားသည်; simulation
  အမြန်နှုန်းသည် သင့်စက်ပေါ် မူတည်ပြီး real-time အာမခံချက် မဟုတ်ပါ။ Native
  viewer ၏ pause/speed control များသည် ဤ loop ကို ထိန်းချုပ်နိုင်ခြင်း မရှိပါ။
- တောင်းဆိုထားသော simulation ကာလ ပြီးဆုံးသည်နှင့် window ပိတ်သွားမည်။ ရပ်တန့်ရန်
  ၎င်းထက်စောပြီး ပိတ်နိုင်သည်; ပြီးစီးသွားသော step များကိုသာ သိမ်းဆည်းမည်။
  ပထမ step မရောက်မီ ပိတ်လိုက်ပါက message ပြပြီး output ကို ကျော်သွားမည်။
- ကြည့်ရှုနေစဉ် recording လုပ်ရန် `--no-video` အစား
  `--video data/runs/gui_run.mp4` ကို အစားထိုးပါ။ ဗီဒီယိုသည် frame များကို
  memory ထဲ buffer လုပ်သောကြောင့် recording run များကို တိုတောင်းစွာထားပါ။

## Docker (Ubuntu 22.04, Python 3.10)

[docker/Dockerfile](docker/Dockerfile) က dependency များကို virtual environment
တစ်ခု ဖန်တီး/ကူးယူခြင်း မရှိဘဲ container ၏ system Python ထဲသို့ `python3 -m pip`
ဖြင့် တိုက်ရိုက် install လုပ်သည်။ Mesa OSMesa က fly ၏ vision အတွက် display server
သို့မဟုတ် GPU passthrough မလိုအပ်ဘဲ CPU-only headless rendering ကို ပံ့ပိုးသည်။
၎င်းသည် GPU rendering ထက် နှေးနိုင်သည်။ Image တူတူပင် local X11/XWayland display
forwarding မှတစ်ဆင့် native MuJoCo GUI ကိုလည်း ပံ့ပိုးသည်။ GUI mode က GLFW/GLX
နှင့် Mesa software rendering ကို သုံးသောကြောင့် GPU passthrough သို့မဟုတ်
NVIDIA toolkit မလိုအပ်ပါ။

### ဒေတာ ပြင်ဆင်ခြင်း

Simulation အပြည့်အစုံ run မလုပ်မီ, [setup guide, section 6](README/01_README.md)
ရှိ ညွှန်ကြားချက်များအတိုင်း FlyWire v783 ဖိုင် နှစ်ခုလုံးကို download လုပ်ပါ:

```text
data/proofread_connections_783.feather
data/proofread_root_ids_783.npy
```

တည်မြဲသော `data/cache/connectome.npz` မရှိလျှင် ဖိုင်နှစ်ခုလုံး လိုအပ်သည်။
Setup guide တွင် link ပေးထားသော dataset ၏ attribution နှင့် CC BY 4.0 license
လိုအပ်ချက်များကို လိုက်နာပါ။ Build လုပ်ခြင်းက ဒေတာကို အလိုအလျောက်
download မလုပ်ပါ။

### Build နှင့် run

Project root မှ:

```bash
bash docker/build_docker_image
bash docker/run_image --help
bash docker/run_image --duration 0.1 --out /data/runs/docker_smoke.npz
bash docker/run_image --duration 1.0 --video /data/runs/run_1s.mp4
```

Helper များသည် အခြား working directory များမှလည်း အလုပ်လုပ်သည်။ Custom image
tag အတွက် `IMAGE_NAME` ကို သတ်မှတ်ပါ။ Run helper က project ၏ `data/` ကို
`/data` တွင် mount လုပ်ပေးသဖြင့် download လုပ်ထားသော ဒေတာကို ပြန်လည် build
မလုပ်ဘဲ သုံးနိုင်ပြီး cache/result များသည် host ပေါ်တွင် ဆက်လက်တည်ရှိမည်။
Root ပိုင်ဆိုင်သော output ဖိုင်များ မဖြစ်ပေါ်စေရန် host ၏ UID/GID ဖြင့်
run ပေးသည်။

တူညီသော တိုက်ရိုက် Docker command များ:

```bash
docker build -f docker/Dockerfile -t romrobotics/fruit_fly_brain:latest .
docker run --rm \
  --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,source=$(pwd)/data,target=/data" \
  romrobotics/fruit_fly_brain:latest \
  --duration 0.1 --out /data/runs/docker_smoke.npz
```

Image ထဲတွင် local source ဒေတာကိုလည်း `/data` ၌ ထည့်သွင်းထားသည်; `/app/data`
သည် ထို့ဆီသို့ link ဆက်ထားသဖြင့် ရှိနှင့်ပြီးသား CLI ပုံမှန်တန်ဖိုးများ
မပြောင်းလဲဘဲ အလုပ်လုပ်သည်။ Bind mount မပါပါက output များသည် container ထဲ
ကျန်ရှိနေပြီး `--rm` ဖြင့် ဖယ်ရှားသည့်အခါ ပျောက်ဆုံးသွားမည်။ Host ပေါ်တွင်
ဖန်တီးထားသော cache/run ဖိုင်များ၊ local virtual environment များနှင့် secret
များကို build context မှ ထုတ်ပယ်ထားသည်။

### Image ကို စစ်ဆေးခြင်း

```bash
docker run --rm --entrypoint python3 \
  romrobotics/fruit_fly_brain:latest -m pip check
docker run --rm --entrypoint python3 \
  romrobotics/fruit_fly_brain:latest -m pytest tests -q -p no:cacheprovider
```

Test များတွင် FlyGym physics/vision ၏ တိုတောင်းသော headless simulation
တစ်ခု ပါဝင်သည်။ Cache မရရှိနိုင်လျှင် real-connectome test ကို ကျော်သွားမည်;
ထည့်သွင်းလိုပါက simulation run တစ်ခုဖြင့် cache ကို build လုပ်ပြီးနောက်
အထက်ပါအတိုင်း `data/` ကို mount လုပ်ပါ။ CLI သည် Python ၏ thread-pool
shutdown မလုပ်မီ OSMesa context များအပါအဝင် simulation/rendering resource
များကို ပြီးစီးချိန် သို့မဟုတ် မအောင်မြင်ချိန်တွင် အတိအလင်း လွှတ်ပေးသည်။

### Docker ထဲတွင် MuJoCo GUI ကို run ခြင်း

X11 သို့မဟုတ် XWayland ရှိသော Linux desktop တွင် project root မှ run ပါ:

```bash
# Ubuntu: host တွင် xauth မရှိသေးလျှင်သာ install လုပ်ပါ။
sudo apt-get install xauth

bash docker/build_docker_image
bash docker/run_image \
  --gui --duration 30.0 --no-video \
  --out /data/runs/docker_gui.npz
```

Argument မပါဘဲ helper ကို run ပါက ဗီဒီယို recording ပိတ်ထားသော GUI ကို
simulation ၃၀ စက္ကန့်အတွက် ဖွင့်ပေးသည်:

```bash
bash docker/run_image
# သို့မဟုတ် working directory သည် docker/ ဖြစ်နေပါက:
./run_image
```

Argument များကို တိုက်ရိုက် ထည့်သွင်းပါက ထိုအတိုင်း ပေးပို့မည်။ ဥပမာ
`bash docker/run_image --duration 1.0 --no-video` သည် headless run တစ်ခု
အဖြစ်ပင် ဆက်ရှိနေမည်။ ဤ helper ၏ ပုံမှန်တန်ဖိုးက image ၏ တိုက်ရိုက်
`docker run` ပုံမှန်တန်ဖိုးများကို မပြောင်းလဲစေပါ။

Helper သည် `--gui` (argument မပါသော ပုံမှန်အခြေအနေအပါအဝင်) ကို သိရှိပြီး
အလိုအလျောက်:

- သင့် local `DISPLAY` ကို ပေးပို့ပြီး X11 socket များကို read-only ဖြင့်
  mount လုပ်သည်;
- ရွေးချယ်ထားသော display ၏ cookie တစ်ခုတည်းပါသည့် mode-600 Xauthority
  ယာယီဖိုင်တစ်ခု ဖန်တီးပြီး read-only ဖြင့် mount လုပ်ကာ Docker ပြီးဆုံးသည့်
  အခါ ဖယ်ရှားသည်;
- Image ၏ headless ပုံမှန်တန်ဖိုးများကို `MUJOCO_GL=glfw` နှင့်
  `PYOPENGL_PLATFORM=glx` ဖြင့် override လုပ်သည်;
- `LIBGL_ALWAYS_SOFTWARE=1` ဖြင့် Mesa software rendering ကို ဖွင့်ပေးသည်;
- သင့် host UID/GID အဖြစ် run ပြီး ရလဒ်များကို host ၏ `data/runs/` တွင်
  ထားရှိသည်။

`xhost +`၊ `--privileged`၊ host networking သို့မဟုတ် GPU device access
မည်သည့်တစ်ခုမျှ မလိုအပ်ပါ။ ယုံကြည်ရသော image များကိုသာ run ပါ: X11 access
သည် application တစ်ခုအား သင့် desktop နှင့် interact လုပ်ခွင့် ပေးသည်။
ယာယီ authentication ဖိုင်ကို ပိတ်ချိန်တွင် ရှင်းလင်းနိုင်စေရန် GUI helper
ကို foreground တွင် သုံးပါ။

GUI နှင့် ဗီဒီယို အတူတကွအတွက် recording ကို တိုတောင်းစွာထားပါ:

```bash
bash docker/run_image \
  --gui --duration 1.0 \
  --out /data/runs/docker_gui_1s.npz \
  --video /data/runs/docker_gui_1s.mp4
```

Camera control များ၊ စောစီးစွာပိတ်ခြင်း အပြုအမူနှင့် ကာလကန့်သတ်ချက်များသည်
အထက်တွင် ဖော်ပြထားသော host GUI အတိုင်း တူညီသည်။ Software rendering နှင့်
whole-brain simulation သည် real-time ထက် နှေးနိုင်သည်။ Headless mode တွင်
interactive GUI မရှိပါ; မူလ OSMesa ဂဏန်း/MP4 workflow ကို ဆက်လက်ထားလိုပါက
`--gui` မပါဘဲ argument များကို တိုက်ရိုက် ပေးပို့ပါ။

#### Display ပြဿနာရှာဖွေခြင်း

- သင့် desktop တွင် ဖွင့်ထားသော terminal တွင် `echo "$DISPLAY"` နှင့်
  `echo "$XAUTHORITY"` ကို စစ်ဆေးပါ။ Helper သည် `:0`၊ `:1`၊ `unix:0`
  ကဲ့သို့သော local display များကိုသာ လက်ခံပြီး `localhost:10.0` ကဲ့သို့
  SSH-forward/TCP display များကို လက်မခံပါ။
- `XAUTHORITY` သတ်မှတ်မထားလျှင် helper က `$HOME/.Xauthority` ကို သုံးမည်။
  အခြားနေရာတွင် သိမ်းဆည်းသော desktop (ဥပမာ GDM) အတွက် helper ကို မခေါ်မီ
  ၎င်း session ၏ တကယ်ဖတ်နိုင်သော ဖိုင်ကို `XAUTHORITY` တွင် သတ်မှတ်ပါ။
  ပျောက်ဆုံးနေသော cookie များကို ဖြတ်ကျော်ရန် X server ၏ access control
  ကို ဒုර်ဗလ မလုပ်ပါ။
- Wayland session တစ်ခုအတွက် XWayland နှင့် မှန်ကန်သော `DISPLAY` လိုအပ်သည်။
  Native Wayland-only session များ၊ remote Docker daemon များနှင့် Docker
  Desktop/Windows/macOS display forwarding များကို ဤ helper က
  configure မလုပ်ပါ။
- GUI အတွက် `MUJOCO_GL=egl` သို့မဟုတ် `PYOPENGL_PLATFORM=osmesa` ကို
  မသုံးပါနှင့်။ GLFW/GLX override များ အတူတကွ အကျုံးဝင်စေရန် helper ကို
  သုံးပါ။
