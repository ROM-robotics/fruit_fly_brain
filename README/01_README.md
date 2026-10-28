# Fruit fly brain: FlyWire SNN + FlyGym + MuJoCo

ဤ project သည် **FlyWire brain connectivity ကို အခြေခံထားသော spiking neural network (SNN)** ကို **FlyGym / NeuroMechFly v2 fly body** နှင့် ချိတ်ဆက်ပြီး **MuJoCo physics environment** ထဲတွင် လမ်းလျှောက်စမ်းသပ်ထားသည့် prototype ဖြစ်ပါတယ်။

**လက်ရှိအခြေအနေ**

- MuJoCo ထဲက fly body နှင့် flat arena ကို အသုံးပြုထားပါတယ်။
- မျက်စိနှစ်ဖက်မှ brightness ကို SNN ထဲ ထည့်ပြီး motor drive ဘယ်/ညာကို တွက်ပါတယ်။
- ခြေထောက်တစ်ချောင်းချင်းကို FlyGym ၏ ရှိပြီးသား walking controller က ထိန်းပါတယ်။
- **Odyssey-3 သို့မဟုတ် အခြား generative world model မချိတ်ထားသေးပါ။**
- CLI သည် default headless ဖြစ်ပြီး optional `.npz` output၊ `--video` ဖြင့် MP4၊ `--gui` ဖြင့် native MuJoCo viewer ကို သုံးနိုင်ပါတယ်။ [MP4](../README.md#mp4-preview-no-gui-required) နှင့် [host GUI အသုံးပြုနည်း](../README.md#native-mujoco-gui-host-desktop) ကို ကြည့်ပါ။
- Biological fly တစ်ကောင်လုံး၏ brain/body ကို အတိအကျ ပြန်လည်ဖန်တီးထားခြင်း၊ Odyssey demo ကို အတိအကျ ပြန်တည်ဆောက်ထားခြင်း မဟုတ်ပါ။

## ၁။ Simulator၊ body၊ brain တို့ကို ဘယ်က ယူထားသလဲ

| အစိတ်အပိုင်း | ရင်းမြစ် | ဤ project မှာ အသုံးပြုပုံ |
| --- | --- | --- |
| Physics simulator | [MuJoCo](https://mujoco.org/) | Gravity၊ joint dynamics၊ collision၊ ground contact နှင့် adhesion ပါသော body ကို simulate လုပ်ပါတယ်။ |
| Simulation interface | [FlyGym 1.2.1](https://github.com/NeLy-EPFL/flygym/tree/v1.2.1)၊ EPFL NeLy Lab | Python မှ fly၊ arena၊ eyes၊ sensors နှင့် controller ကို ဖန်တီးပါတယ်။ MuJoCo နှင့် `dm_control` ကို dependency အဖြစ် install လုပ်ပေးပါတယ်။ |
| Fly body | FlyGym ထဲပါသော **NeuroMechFly v2** | Adult female fruit fly ၏ micro-CT scan ကို အခြေခံထားသော biomechanical model ဖြစ်ပါတယ်။ Model ကို ကိုယ်တိုင်ဆွဲထားခြင်း မဟုတ်ပါ။ |
| Brain wiring | [FlyWire whole-brain connectivity data, Zenodo record 10676866](https://doi.org/10.5281/zenodo.10676866) | v783 proofread root IDs နှင့် synaptic connections ကို download လုပ်ထားပါတယ်။ |
| Brain dynamics | ဤ project ၏ [lif.py](./lif.py) | NumPy + SciPy sparse matrix ဖြင့် ရေးထားသော leaky integrate-and-fire (LIF) network ဖြစ်ပါတယ်။ |
| Sensory/motor mapping | ဤ project ၏ [connectome.py](./connectome.py) နှင့် [closed_loop.py](./closed_loop.py) | Optic-neuropil input pools နှင့် GNG-output motor proxy pools ကို heuristic ဖြင့် ရွေးပြီး ချိတ်ထားပါတယ်။ |

### Simulator နှင့် body ကို ယူထားပုံ

[requirements.txt](../requirements.txt) မှ `flygym==1.2.1` ကို install လုပ်သည့်အခါ Python package နှင့်အတူ body အတွက် MJCF XML၊ mesh နှင့် အခြား model assets တွေပါ လာပါတယ်။ Blender model သီးသန့် download လုပ်ရန်၊ MuJoCo application သီးသန့် ဖွင့်ရန် မလိုပါ။

လက်ရှိ [world.py](./world.py) က အောက်ပါအတိုင်း သုံးထားပါတယ်။

```python
from flygym import Fly

fly = Fly(
    enable_adhesion=True,
    enable_vision=True,
    contact_sensor_placements=contact_sensors,
)
```

ဤ fragment ထဲက `contact_sensors` ကို [world.py](./world.py) မှာ သတ်မှတ်ထားပါတယ်။ Fragment တစ်ခုတည်းကို standalone run ရန် မဟုတ်ပါ။

`xml_variant` ကို override မလုပ်ထားသောကြောင့် FlyGym 1.2.1 ၏ default `seqik` body model ကို သုံးပါတယ်။ Installed package အတွင်းရှိ ဆက်စပ်နေရာများမှာ—

```text
flygym/
├── config.yaml
└── data/
    └── mjcf/
        └── neuromechfly_seqik_kinorder_ypr.xml
```

မိမိ install ထားသော package နှင့် XML နေရာကို ကြည့်ရန်၊ project root မှ—

```bash
python - <<'PY'
from pathlib import Path
import flygym

package = Path(flygym.__file__).resolve().parent
print("FlyGym package:", package)
print("Body XML:", package / "data/mjcf/neuromechfly_seqik_kinorder_ypr.xml")
PY
```

### Brain ကို ယူထားပုံ

**Brain model အဆင်သင့် သို့မဟုတ် trained neural-network weights ကို download လုပ်ထားတာ မဟုတ်ပါ။** Download လုပ်ထားတာက neuron IDs နှင့် anatomical connectivity data ဖြစ်ပါတယ်။

1. FlyWire v783 proofread neuron IDs ကို load လုပ်ပါတယ်။
2. Pre/post neuron IDs၊ synapse counts နှင့် neurotransmitter probability columns များကို ဖတ်ပါတယ်။
3. Connection များကို signed sparse matrix အဖြစ် ပြောင်းပါတယ်။
4. ထို wiring ပေါ်မှာ ကျွန်ုပ်တို့ရေးထားသော LIF dynamics ကို run ပါတယ်။
5. ရွေးထားသော neuron pools မှ activity ကို FlyGym drive အဖြစ် ပြောင်းပါတယ်။

**Body model နှင့် FlyWire brain data သည် သီးခြားရင်းမြစ်နှစ်ခုမှ ဖြစ်ပြီး တစ်ကောင်တည်းကနေ scan လုပ်ထားသော body–brain အတွဲ မဟုတ်ပါ။**

## ၂။ Closed-loop က ဘယ်လို အလုပ်လုပ်သလဲ

```text
MuJoCo flat arena + NeuroMechFly body
              |
              v
FlyGym compound-eye vision
              |
              v
မျက်စိတစ်ဖက်စီ၏ mean brightness [left, right]
              |
              v
FlyWire optic-neuropil input pools
              |
              v
NumPy LIF SNN + signed sparse connectivity
              |
              v
GNG-output proxy pools ၏ ဘယ်/ညာ spike rates
              |
              v
Descending drive [left, right]
              |
              v
FlyGym HybridTurningController
              |
              v
Joint positions + leg adhesion -> MuJoCo -> vision အသစ်
```

- Physics timestep သည် **0.0001 s = 0.1 ms** ဖြစ်ပါတယ်။
- Physics **10 steps တိုင်း** SNN ကို တစ်ကြိမ် update လုပ်ပါတယ်။ SNN timestep သည် **1 ms** ဖြစ်ပါတယ်။
- FlyGym ၏ default vision refresh rate သည် **500 Hz** ဖြစ်ပါတယ်။ Physics step တိုင်း image အသစ် render လုပ်သည်ဟု မဆိုလိုပါ။
- FlyGym vision observation သည် မျက်စိ ၂ ဖက်၊ တစ်ဖက်လျှင် ommatidia 721 ခု၊ channel ၂ ခု ပါသော `(2, 721, 2)` array ဖြစ်ပါတယ်။
- လက်ရှိ encoder က ommatidia/channel များကို average ယူပြီး brightness ၂ ခုသာ သုံးပါတယ်။ Image shape၊ object recognition၊ optic-flow processing မပါသေးပါ။
- Contact sensors ကို walking controller က အသုံးပြုပါတယ်။ Joint/contact feedback ကို SNN ထဲ တိုက်ရိုက် မပို့ထားသေးပါ။
- Flight မဟုတ်ဘဲ **walking simulation** ဖြစ်ပါတယ်။ Olfaction ကိုလည်း လက်ရှိ wrapper မှာ မဖွင့်ထားပါ။

### Connectivity နှင့် mapping ဆိုင်ရာ assumptions

[connectome.py](./connectome.py) တွင် signed weight ကို—

```text
raw_weight = syn_count × (ach_avg − gaba_avg − glut_avg)
```

ဟု သတ်မှတ်ပါတယ်။ ACh ကို excitatory၊ GABA နှင့် glutamate ကို inhibitory ဟု ယူဆထားတာ ဖြစ်ပါတယ်။ Octopamine၊ serotonin နှင့် dopamine effects မပါပါ။ ဤ sign convention သည် simplified modeling assumption ဖြစ်ပြီး biological connection တစ်ခုစီ၏ effect ကို အပြည့်အဝ အတည်ပြုထားခြင်း မဟုတ်ပါ။

- `weights[post, pre]` ဖြစ်သောကြောင့် `weights @ spikes` က postsynaptic input ကို ထုတ်ပါတယ်။
- Root-ID list ထဲ မပါသော neuron နှင့် ဆက်စပ်သည့် edges များကို ဖယ်ပါတယ်။
- Neuron pair တူသော connections များကို sum လုပ်ပြီး signed weight သုညဖြစ်သည့် pair များကို ဖယ်ပါတယ်။
- **Input pools:** `ME_L/LO_L` သို့မဟုတ် `ME_R/LO_R` မှ incoming synapses အနည်းဆုံး 10 ခုရှိပြီး စုစုပေါင်း retained input ၏ အနည်းဆုံး 50% ဖြစ်သော neurons ကို ရွေးပါတယ်။ အမှန်တကယ် photoreceptor identity ကို annotation ဖြင့် ရွေးထားခြင်း မဟုတ်ပါ။
- **Motor proxy pools:** `GNG` သို့ outgoing synapses အနည်းဆုံး 10 ခုရှိပြီး စုစုပေါင်း retained output ၏ အနည်းဆုံး 50% ဖြစ်သော neurons ကို ရွေးပါတယ်။
- Motor candidate ၏ left/right ကို ဘယ်ဘက်/ညာဘက် neuropils မှ input ပိုရသလဲဖြင့် ခန့်မှန်းပါတယ်။ Verified descending-neuron သို့မဟုတ် leg motor-neuron mapping မဟုတ်ပါ။

[lif.py](./lif.py) တွင် nonzero absolute weights ၏ median ကို `synapse_mv` နှင့် ညီအောင် globally rescale လုပ်ပါတယ်။ Neuron တစ်ခုစီကို uniform LIF parameters သုံးထားပြီး learning/plasticity မပါပါ။

Motor drive ကို [closed_loop.py](./closed_loop.py) တွင်—

```text
external_input = input_gain_mv × eye_brightness
motor_rate_hz = pool အတွင်း လက်ရှိ step မှာ fire လုပ်သည့် fraction / SNN timestep
drive = base_drive + motor_gain × clip(motor_rate_hz / max_motor_rate_hz, 0, 1)
```

ဟု တွက်ပါတယ်။ Motor rate သည် 1 ms window အတွင်း pool-average estimate ဖြစ်ပြီး temporal smoothing မပါပါ။

**SNN မပစ်ခတ်လည်း `base_drive=0.8` ရှိသောကြောင့် fly သည် လမ်းလျှောက်နိုင်ပါတယ်။** ထို့ကြောင့် displacement ရှိရုံဖြင့် connectome က biological behavior ကို မှန်ကန်စွာ ထိန်းနိုင်ပြီဟု မသတ်မှတ်နိုင်ပါ။

## ၃။ Project ဖိုင်များ

အောက်ပါ commands အားလုံးကို **project root** မှ run ပါ။ README ရှိသော package folder အတွင်းမှ မဟုတ်ပါ။

```text
/home/mr_robot/Desktop/fruit_fly_brain/    <- project root
├── requirements.txt
├── Copilot.md
├── .venv/
├── fruitfly_brain/                      <- Python package
│   ├── README.md
│   ├── __init__.py
│   ├── connectome.py
│   ├── lif.py
│   ├── world.py
│   └── closed_loop.py
├── tests/
└── data/
    ├── proofread_connections_783.feather
    ├── proofread_root_ids_783.npy
    ├── cache/
    │   └── connectome.npz
    └── runs/
        └── final_1s.npz
```

| ဖိုင် | တာဝန် |
| --- | --- |
| [connectome.py](./connectome.py) | FlyWire loading၊ signed matrix၊ sensory/motor pools နှင့် cache |
| [lif.py](./lif.py) | LIF parameters၊ membrane dynamics၊ spikes နှင့် refractory state |
| [world.py](./world.py) | FlyGym body၊ sensors၊ arena/controller wrapper နှင့် Odyssey placeholder |
| [closed_loop.py](./closed_loop.py) | Brain/body loop၊ CLI နှင့် result saving |
| [requirements.txt](../requirements.txt) | Direct dependencies နှင့် pinned versions |
| [tests/](../tests/) | Synthetic connectivity tests၊ LIF tests နှင့် FlyGym smoke test |
| [Copilot.md](../Copilot.md) | Odyssey demo ဆိုင်ရာ အစောပိုင်း ရှင်းလင်းချက်နှင့် background |

## ၄။ လက်ရှိစက်မှာ ချက်ချင်း run ရန်

ယခု workspace တွင် [virtual environment](../.venv/)၊ [FlyWire data](../data/) နှင့် [connectome cache](../data/cache/connectome.npz) ရှိပြီးသားဖြစ်ပါတယ်။ ပြန် install/download မလုပ်ဘဲ စမ်းနိုင်ပါတယ်။

```bash
cd /home/mr_robot/Desktop/fruit_fly_brain

env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python \
  -m fruitfly_brain.closed_loop \
  --duration 1.0 \
  --out data/runs/my_first_run.npz
```

- `env -u PYTHONPATH` သည် ယခု command အတွက်သာ inherited Python search path ကို ဖယ်ပါတယ်။ လက်ရှိစက်၏ ROS packages ဝင်မရောစေရန် ဖြစ်ပါတယ်။
- `MUJOCO_GL=egl` သည် offscreen rendering backend ကို ရွေးပါတယ်။ Window မဖွင့်သော်လည်း fly vision အတွက် rendering လိုပါတယ်။
- `--duration 1.0` သည် **simulated time 1 second** ဖြစ်ပါတယ်။ Wall-clock time 1 second အတွင်း ပြီးမည်ဟု မဆိုလိုပါ။
- `--out` ရှိမှ result ကို သိမ်းပါတယ်။ နာမည်တူ output ရှိပြီးသားဆို overwrite ဖြစ်သောကြောင့် run အသစ်များအတွက် နာမည်အသစ် သုံးပါ။

## ၅။ Environment အသစ်တွင် install လုပ်ရန်

### လိုအပ်ချက်များ

- လက်ရှိ စမ်းသပ်ထားသော environment သည် **Linux + Python 3.10.12** ဖြစ်ပါတယ်။
- FlyGym 1.2.1 package metadata က Python `>=3.10,<3.13` လိုအပ်ပါတယ်။ ဤ project ၏ dependency set အပြည့်ကိုတော့ Python 3.10 တွင်သာ စမ်းသပ်ထားပါတယ်။
- CPU ဖြင့် SNN run ပါတယ်။ CUDA၊ PyTorch သို့မဟုတ် model training မလိုပါ။
- FlyGym vision အတွက် working OpenGL/EGL driver လိုပါတယ်။ CUDA မလိုခြင်းသည် rendering backend မလိုဟု မဆိုလိုပါ။
- First build က RAM အချို့ GB သုံးနိုင်ပါတယ်။ လက်ရှိစမ်းသပ်မှုတွင် 2–3 GB ဝန်းကျင် သုံးခဲ့သော်လည်း စက်နှင့် library version အလိုက် ကွာနိုင်ပါတယ်။ RAM 8 GB နှင့်အထက်၊ disk space အနည်းဆုံး several GB လွတ်ထားရန် အကြံပြုပါတယ်။
- Dependencies နှင့် source data မရှိသေးလျှင် download အတွက် internet လိုပါတယ်။

Project source folder ကို စက်အသစ်သို့ ယူပြီးနောက်—

```bash
cd /path/to/fruit_fly_brain

env -u PYTHONPATH python3.10 -m venv .venv
source .venv/bin/activate

env -u PYTHONPATH python -m pip install -r requirements.txt
```

`/path/to/fruit_fly_brain` ကို မိမိ project root နေရာဖြင့် အစားထိုးပါ။ အခြားစက်မှ `.venv` ကို copy ကူးသုံးခြင်းထက် စက်အသစ်တွင် venv အသစ်ဆောက်ပါ။ လက်ရှိစက်မှာတော့ ရှိပြီးသား venv ကို ပြန်ဆောက်ရန် မလိုပါ။

### Git clone ပြီး `.venv` ပြန်ဆောက်ရန်

**`.venv` သည် project ကို local run ဖို့ လိုသော်လည်း Git ထဲ commit လုပ်ရန် မလိုပါ။** ထို folder တွင် installed packages၊ executables နှင့် စက်ပေါ်ရှိ absolute paths များ ပါနိုင်သောကြောင့် စက်တစ်လုံးမှတစ်လုံး ကူးသုံးရန် မသင့်ပါ။

Project root ၏ [.gitignore](../.gitignore) က root နှင့် subdirectories အတွင်းရှိ `.venv`၊ `.vscode`၊ `__pycache__` နှင့် tool caches များကို ignore လုပ်ထားပါတယ်။ Ignore လုပ်ခြင်းသည် local files များကို ဖျက်ခြင်း မဟုတ်ပါ။ Clone လုပ်သည့်အခါ `.venv` ကို auto-download/auto-create မလုပ်ပေးပါ။

Remote repository ရှိပြီး source ကို push လုပ်ထားပြီးမှ အောက်ပါအတိုင်း clone/install လုပ်နိုင်ပါတယ်။ `REPOSITORY_URL` ကို မိမိ remote URL ဖြင့် အစားထိုးပါ။

```bash
git clone REPOSITORY_URL fruit_fly_brain
cd fruit_fly_brain

env -u PYTHONPATH python3.10 -m venv .venv
source .venv/bin/activate
env -u PYTHONPATH python -m pip install -r requirements.txt
```

ဤနည်းဖြင့် [.venv](../.venv/) ကို clone လုပ်ထားသောစက်ပေါ်တွင် အသစ်ဖန်တီးပြီး [requirements.txt](../requirements.txt) အတိုင်း dependencies ကို install လုပ်ပါတယ်။ Requirements သည် full transitive lockfile မဟုတ်သောကြောင့် environment ကို byte-for-byte တူညီအောင် ပြန်ယူခြင်း မဟုတ်ပါ။

ပြန်ဆောက်၍ရသော [connectome cache](../data/cache/) နှင့် [run outputs](../data/runs/) ကိုလည်း ignore လုပ်ထားပါတယ်။ ထို့အပြင် download ပြန်လုပ်နိုင်သော raw FlyWire ဖိုင်နှစ်ဖိုင်ကို ignore လုပ်ထားပြီး connection file တစ်ဖိုင်တည်း ~852 MB ရှိသောကြောင့် ပုံမှန် Git repository ထဲ မထည့်ထားသင့်ပါ။ Clone ပြီးနောက် အောက်ပါ data-download section အတိုင်း ဖိုင်နှစ်ဖိုင်ကို ရယူပါ။ ပထမ simulation run မှ cache ကို ဆောက်ပေးပါမယ်။

**Tracked files ဆိုင်ရာ မှတ်ချက်:** `.gitignore` သည် Git က track လုပ်ထားပြီးသား files များကို untrack မလုပ်ပေးပါ။ ထိုသို့ရှိလျှင် မိမိလိုအပ်သော specific paths များကို အရင်စစ်ပြီး `git rm --cached` ဖြင့် index မှသာ ဖယ်ရန် လိုပါတယ်။ ဤ setup တွင် local environment/data ကို မဖျက်ထားသလို existing tracked files ကိုလည်း အလိုအလျောက် untrack မလုပ်ထားပါ။

`venv` module မရှိသည့် Ubuntu 22.04 စက်တွင်သာ system prerequisite အဖြစ်—

```bash
sudo apt-get update
sudo apt-get install python3.10-venv
```

ဤ project တွင် `pyproject.toml`/`setup.py` မရှိသေးသောကြောင့် `pip install -e .` ဖြင့် install လုပ်သည့် package မဟုတ်ပါ။ Project root မှ `python -m fruitfly_brain.closed_loop` ဖြင့် run ပါ။

### Installed versions စစ်ရန်

```bash
env -u PYTHONPATH python - <<'PY'
import sys
from importlib.metadata import version

print("Python:", sys.version.split()[0])
for name in ("flygym", "mujoco", "dm-control", "numpy", "pandas", "pyarrow", "scipy"):
    print(name, version(name))
PY
```

လက်ရှိ workspace တွင် အတည်ပြုထားသော versions—

| Package | Version |
| --- | --- |
| FlyGym | 1.2.1 |
| MuJoCo | 3.2.7 |
| dm-control | 1.0.27 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| PyArrow | 25.0.1 |
| SciPy | 1.15.3 |

MuJoCo နှင့် dm-control versions ကို FlyGym 1.2.1 က pin လုပ်ထားပါတယ်။ [requirements.txt](../requirements.txt) သည် full transitive lockfile မဟုတ်သောကြောင့် dependency အားလုံးကို စက်တိုင်းတွင် တစ်ပုံစံတည်း resolve လုပ်မည်ဟု အာမခံမထားပါ။

## ၆။ FlyWire data ကို download လုပ်ရန်

**လက်ရှိ workspace တွင် နှစ်ဖိုင်စလုံး ရှိပြီး checksum ကို စစ်ပြီးသားဖြစ်ပါတယ်။** စက်အသစ်တွင် မရှိမှ download လုပ်ပါ။ CLI က internet မှ auto-download မလုပ်ပါ။

ရင်းမြစ်: [FlyWire Whole-brain Connectome Connectivity Data](https://doi.org/10.5281/zenodo.10676866)၊ license **CC BY 4.0** ဖြစ်ပါတယ်။ ပြန်လည်အသုံးပြု/ဖြန့်ဝေရာတွင် dataset attribution နှင့် license terms ကို လိုက်နာပါ။

Project root မှ—

```bash
mkdir -p data

curl -fL --retry 3 -C - \
  -o data/proofread_connections_783.feather \
  'https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content'

curl -fL --retry 3 -C - \
  -o data/proofread_root_ids_783.npy \
  'https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content'
```

`-C -` သည် interrupted partial download ကို resume လုပ်ရန် ဖြစ်ပါတယ်။ Connection file က ကြီးသောကြောင့် download ပြီးမှ checksum စစ်ပြီး run ပါ။ Download မပြီးသော `.feather` ဖိုင်ကို loader က ဖတ်၍ မရပါ။

| ဖိုင် | Exact size | Zenodo MD5 |
| --- | --- | --- |
| [proofread_connections_783.feather](../data/proofread_connections_783.feather) | 852,022,274 bytes၊ ~852 MB | `f48f972d262323a102aed49af1396b8a` |
| [proofread_root_ids_783.npy](../data/proofread_root_ids_783.npy) | 1,114,168 bytes၊ ~1.1 MB | `e0e6c19732fd8c7a4e39a2d170105421` |

Checksum စစ်ရန်—

```bash
md5sum data/proofread_connections_783.feather data/proofread_root_ids_783.npy
```

Expected output—

```text
f48f972d262323a102aed49af1396b8a  data/proofread_connections_783.feather
e0e6c19732fd8c7a4e39a2d170105421  data/proofread_root_ids_783.npy
```

ဤ checksum ကို transfer integrity စစ်ရန် သုံးထားတာဖြစ်ပါတယ်။

## ၇။ Simulation run ရန်နှင့် CLI options

Venv ကို activate လုပ်ထားပြီး project root မှ—

```bash
env -u PYTHONPATH MUJOCO_GL=egl python \
  -m fruitfly_brain.closed_loop \
  --duration 1.0 \
  --out data/runs/run_1s.npz
```

ပိုရှည်သော run အတွက်—

```bash
env -u PYTHONPATH MUJOCO_GL=egl python \
  -m fruitfly_brain.closed_loop \
  --duration 2.0 \
  --out data/runs/run_2s.npz
```

နှစ်ဖိုင်နှင့် cache တည်နေရာကို ကိုယ်တိုင်ပေးလိုလျှင်—

```bash
env -u PYTHONPATH MUJOCO_GL=egl python \
  -m fruitfly_brain.closed_loop \
  --duration 1.0 \
  --edges data/proofread_connections_783.feather \
  --root-ids data/proofread_root_ids_783.npy \
  --cache data/cache/connectome.npz \
  --out data/runs/custom_run.npz
```

Help ကြည့်ရန်—

```bash
python -m fruitfly_brain.closed_loop --help
```

| Option | Default | အဓိပ္ပာယ် |
| --- | --- | --- |
| `--duration` | `1.0` | Simulated seconds၊ positive value ပေးပါ။ Zero/negative/tiny-duration edge cases အတွက် CLI validation မပြည့်စုံသေးပါ။ |
| `--edges` | Project root အောက်က `data/proofread_connections_783.feather` | Raw connectivity data |
| `--root-ids` | Project root အောက်က `data/proofread_root_ids_783.npy` | Proofread neuron IDs |
| `--cache` | Project root အောက်က `data/cache/connectome.npz` | Cached sparse matrix နှင့် pools |
| `--out` | မရှိ | ပေးထားမှ run arrays ကို `.npz` အဖြစ် သိမ်းမည် |
| `--gui` | ပိတ်ထား | Host desktop မှာ native MuJoCo viewer ဖွင့်မည်။ `MUJOCO_GL=glfw` လိုပြီး `PYOPENGL_PLATFORM` ကို unset လုပ်ပါ။ |
| `--video [MP4]` | `SAVE_VIDEO=False` ဆို မရှိ | MP4 record လုပ်မည်။ Path မပေးလျှင် project root အောက်က `data/runs/simulation.mp4` ကို သုံးမည်။ |
| `--no-video` | မပေးထား | Global `SAVE_VIDEO=True` ဖြစ်လည်း video ပိတ်မည်။ |

[closed_loop.py](../fruitfly_brain/closed_loop.py) တွင် `SAVE_VIDEO = True`
ထားလျှင် ordinary CLI runs မှာ MP4 ကို အလိုအလျောက် သိမ်းပါတယ်။
Default `False` ဖြစ်သောကြောင့် မူလ numerical-only behavior မပြောင်းပါ။
Video သည် 640×480၊ 30 FPS၊ 0.2× playback ဖြစ်ပြီး simulated 1 s ကို
ခန့်မှန်း 5 s ကြာ ကြည့်နိုင်ပါတယ်။ Frame buffering ကြောင့် memory ပိုသုံးပါတယ်။

Default data/cache paths များသည် source file မှတွက်သော project root ကို အခြေခံပါတယ်။ ကိုယ်တိုင်ပေးသော relative paths များ၊ ဥပမာ `--out data/runs/run_1s.npz` သည် လက်ရှိ working directory ကို အခြေခံပါတယ်။

### Cache က ဘယ်လို အလုပ်လုပ်သလဲ

- Cache မရှိလျှင် raw data နှစ်ဖိုင်မှ matrix နှင့် pools ကို ဆောက်ပြီး cache သိမ်းပါတယ်။
- Cache ရှိလျှင် raw data မဖတ်ဘဲ cache ကို တိုက်ရိုက် load ပါတယ်။
- Cache အတွင်း signed raw matrix နှင့် neuron pools ကို သိမ်းပါတယ်။ LIF ၏ median-weight scaling ကို network တည်ဆောက်ချိန်မှ လုပ်ပါတယ်။
- Cache သည် ယခု workspace တွင် ~78 MB ဖြစ်ပါတယ်။ First build သည် ဤစက်၏ အစောပိုင်းစမ်းသပ်မှုတွင် ~16 seconds ကြာခဲ့ပြီး guaranteed timing မဟုတ်ပါ။
- **Raw data၊ pool-selection logic သို့မဟုတ် signed-weight formula ပြောင်းလဲလျှင် cache ကို auto-invalidate မလုပ်ပါ။** `--cache data/cache/connectome_rebuilt.npz` ကဲ့သို့ **မရှိသေးသော cache path အသစ်** ပေးပြီး ပြန်ဆောက်ပါ။
- Corrupt cache ရှိလျှင် အလိုအလျောက် raw data ဖြင့် fallback မလုပ်ဘဲ error တက်ပါမယ်။

### Expected terminal output

2026-10-24 တွင် သိမ်းထားသော [1-second run](../data/runs/final_1s.npz) ၏ summary—

```text
neurons=139255 synapse_pairs=9465294 input L/R=32143/32195 motor L/R=529/542
steps=10000 total_spikes=2707835 mean_drive=[1.003, 0.992] displacement_mm=13.761
saved data/runs/final_1s.npz
```

`saved` line သည် မိမိပေးသော output path အတိုင်း ပြောင်းပါမယ်။ စက်၊ driver နှင့် dependencies ကြောင့် numeric results အတိအကျ တူမည်ဟု မအာမခံပါ။

- `neurons` = root-ID list ထဲရှိ 139,255 neurons။
- `synapse_pairs` = aggregation/sign cancellation ပြီးနောက် nonzero **directed neuron pairs** 9,465,294 ခု။ Individual biological synapses အရေအတွက် မဟုတ်ပါ။
- `input L/R`၊ `motor L/R` = heuristic pools ထဲရှိ neuron အရေအတွက်။
- `steps` = physics steps။ 1 second တွင် 10,000 steps ဖြစ်ပါတယ်။
- `total_spikes` = neurons အားလုံး၊ SNN steps အားလုံးပေါင်း spike events။ Neuron အရေအတွက် မဟုတ်ပါ။
- `mean_drive` = SNN steps တစ်လျှောက် ဘယ်/ညာ descending drive ပျမ်းမျှ။
- `displacement_mm` = ပထမ record position မှ နောက်ဆုံး position သို့ **3D straight-line distance**။ လျှောက်ခဲ့သမျှ path length မဟုတ်သလို vertical movement လည်း ပါနိုင်ပါတယ်။

Short smoke run ဖြစ်သော 0.01 second တွင် spikes မထွက်သေးနိုင်ပါ။ ထိုအချိန် fly ရွေ့နေခြင်းသည် base drive နှင့် physics ကြောင့် ဖြစ်နိုင်ပါတယ်။ Brain activity ကို ကြည့်ရန် 1-second run ကို စတင်သုံးပါ။

## ၈။ Output `.npz` ကို ဖတ်ရန်

`.npz` သည် video မဟုတ်ဘဲ NumPy arrays container ဖြစ်ပါတယ်။

1-second run အတွက်—

| Key | Shape | အကြောင်းအရာ |
| --- | --- | --- |
| `positions` | `(10000, 3)` | Physics step တိုင်း၏ x/y/z position၊ FlyGym units အရ mm |
| `drives` | `(1000, 2)` | SNN update တိုင်း၏ left/right drive |
| `motor_rates_hz` | `(1000, 2)` | Motor pool-average instantaneous rates |
| `spike_counts` | `(1000,)` | SNN update တိုင်း၏ total spike count |

အထက်က quick-start run ကို ဖတ်ရန်—

```bash
python - <<'PY'
import numpy as np

with np.load("data/runs/my_first_run.npz") as run:
    for name in run.files:
        print(name, run[name].shape, run[name].dtype)

    positions = run["positions"]
    print("Total spikes:", int(run["spike_counts"].sum()))
    print("Mean drive L/R:", run["drives"].mean(axis=0))
    print("Mean motor rate L/R (Hz):", run["motor_rates_hz"].mean(axis=0))
    print("Displacement (mm):", np.linalg.norm(positions[-1] - positions[0]))
PY
```

Quick-start run မလုပ်ရသေးလျှင် အထက်ပါ snippet ၏ input path ကို ရှိပြီးသား `data/runs/final_1s.npz` ဖြင့် အစားထိုးနိုင်ပါတယ်။

**GUI/video ဆိုင်ရာ မှတ်ချက်:** [world.py](./world.py) တွင် `render()` wrapper နှင့် camera ရှိသော်လည်း လက်ရှိ `run_closed_loop()` က `world.render()` ကို မခေါ်ပါ။ CLI တွင် `--render`၊ `--video` သို့မဟုတ် interactive MuJoCo viewer option မရှိသေးပါ။ ကြည့်ရှုနိုင်သော video/viewer လိုလျှင် render loop နှင့် frame saving/viewer integration ကို ထပ်ရေးရန် လိုပါတယ်။

## ၉။ Configuration ပြောင်းရန်

လက်ရှိ CLI က gain parameters ကို options အဖြစ် မပေးထားပါ။ Default များကို ပြောင်းလိုလျှင် [LoopConfig](./closed_loop.py) နှင့် [LIFConfig](./lif.py) တွင် ပြောင်းနိုင်သလို Python မှ config instance ပေး၍လည်း run နိုင်ပါတယ်။

| Config | Default | အဓိပ္ပာယ် |
| --- | --- | --- |
| `LoopConfig.brain_every` | `10` | Physics steps per SNN update |
| `LoopConfig.input_gain_mv` | `3.0` | Brightness တစ် unit အတွက် SNN step တိုင်းထည့်သော input၊ mV |
| `LoopConfig.base_drive` | `0.8` | SNN activity မရှိလည်း ထည့်ထားသော motor drive |
| `LoopConfig.motor_gain` | `0.6` | SNN activity မှ ထပ်ပေါင်းနိုင်သော drive |
| `LoopConfig.max_motor_rate_hz` | `100.0` | Motor gain အပြည့်ရောက်သော pool rate |
| `LIFConfig.dt_ms` | `1.0` | SNN timestep |
| `LIFConfig.tau_m_ms` | `20.0` | Membrane time constant |
| `LIFConfig.v_rest_mv` / `v_reset_mv` | `-65.0` | Rest/reset potentials |
| `LIFConfig.v_thresh_mv` | `-50.0` | Spike threshold |
| `LIFConfig.t_ref_ms` | `2.0` | Refractory duration |
| `LIFConfig.synapse_mv` | `1.0` | Median nonzero absolute synapse weight ကို scale လုပ်ပြီးနောက် ရရှိသော mV |

Default drive range သည် ဘယ်/ညာတစ်ဖက်စီအတွက် **0.8 မှ 1.4** ဖြစ်ပါတယ်။ Raw motor rate က 100 Hz ထက်ကျော်နိုင်သော်လည်း decoder က clip လုပ်ပါတယ်။

Python API ဥပမာ၊ venv activate လုပ်ပြီး project root မှ—

```bash
env -u PYTHONPATH MUJOCO_GL=egl python - <<'PY'
from fruitfly_brain.closed_loop import DATA_DIR, LoopConfig, run_closed_loop
from fruitfly_brain.connectome import load_connectome
from fruitfly_brain.lif import LIFConfig, LIFNetwork
from fruitfly_brain.world import FlyGymWorld

conn = load_connectome(
    DATA_DIR / "proofread_connections_783.feather",
    DATA_DIR / "proofread_root_ids_783.npy",
    DATA_DIR / "cache/connectome.npz",
)
cfg = LoopConfig(input_gain_mv=3.0, base_drive=0.8, motor_gain=0.6)
world = FlyGymWorld()
net = LIFNetwork(
    conn.weights,
    LIFConfig(dt_ms=cfg.brain_every * world.timestep * 1000),
)
record = run_closed_loop(conn, net, world, duration_s=1.0, cfg=cfg)
print("Total spikes:", int(record["spike_counts"].sum()))
PY
```

`run_closed_loop()` သည် world ကို reset လုပ်သော်လည်း SNN ကို reset မလုပ်ပါ။ Network တစ်ခုတည်းကို independent experiments အတွက် ပြန်သုံးလျှင် run တစ်ခုစီမတိုင်မီ `net.reset()` ခေါ်ပါ၊ သို့မဟုတ် network အသစ်ဖန်တီးပါ။

SNN timestep နှင့် `brain_every × world.timestep` မတူလျှင် `ValueError` တက်ပါတယ်။ Gain နှင့် synapse strength ကို ပြောင်းရာတွင် silent network သို့မဟုတ် excessive firing ဖြစ်နိုင်သောကြောင့် spike counts၊ motor rates နှင့် drive clipping ကို အတူစစ်ပါ။ လက်ရှိ values များသည် exploratory settings သာဖြစ်ပြီး biological calibration အတည်ပြုထားခြင်း မဟုတ်ပါ။

## ၁၀။ Tests run ရန်

Project root မှ၊ venv activate လုပ်ထားလျှင်—

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  python -m pytest tests -q
```

Venv activate မလုပ်ထားလျှင် `python` အစား `.venv/bin/python` သုံးပါ။

အစောပိုင်း run တွင် **14 tests passed** ဖြစ်ခဲ့ပါတယ်။ Real-connectome test သည် default cache မရှိသေးလျှင် skip ဖြစ်ပါတယ်။

- [test_connectome.py](../tests/test_connectome.py): Weight signs၊ duplicate aggregation၊ matrix orientation၊ heuristic pool selection၊ cache round-trip နှင့် optional real-cache sanity။
- [test_lif.py](../tests/test_lif.py): No-input behavior၊ excitatory propagation၊ inhibition၊ refractory နှင့် reset။
- [test_world.py](../tests/test_world.py): FlyGym eye output/movement နှင့် Odyssey placeholder failure။

World test များသည် rendering-capable environment လိုပါတယ်။ Unit tests ဖြတ်ခြင်းသည် mapping ၏ biological validity သို့မဟုတ် meaningful visual navigation ကို အတည်ပြုခြင်း မဟုတ်ပါ။

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` သည် လက်ရှိစက်တွင် ROS `launch_testing` plugins ဝင်လာပြီး pytest collection fail ဖြစ်ခြင်းကို ရှောင်ရန် ဖြစ်ပါတယ်။ ဤ suite သည် external pytest plugins မလိုပါ။

## ၁၁။ အဖြစ်များသော ပြဿနာများ

### `No module named fruitfly_brain`

Project root မှ run နေကြောင်း စစ်ပါ။ `python fruitfly_brain/closed_loop.py` အစား module entry point ကို သုံးပါ။

```bash
cd /home/mr_robot/Desktop/fruit_fly_brain
.venv/bin/python -m fruitfly_brain.closed_loop --help
```

### `FileNotFoundError` သို့မဟုတ် Feather/Arrow file error

Cache မရှိသေးလျှင် source data နှစ်ဖိုင်လုံး လိုပါတယ်။ Download ပြီးစီးမှုကို checksum ဖြင့် စစ်ပါ။ `--edges`၊ `--root-ids` နှင့် `--cache` paths ကိုလည်း စစ်ပါ။ Data အသစ်ပေးသော်လည်း cache အဟောင်းရှိလျှင် loader က cache အဟောင်းကိုပဲ သုံးပါတယ်။

### EGL / GLFW / OpenGL error

Headless run အတွက် `MUJOCO_GL=egl` ကို Python process မစတင်မီ သတ်မှတ်ပါ။ EGL backend သည် system graphics libraries/driver ပေါ် မူတည်ပါတယ်။

လိုအပ်သော shared library မရှိကြောင်း error တက်သော Ubuntu/Debian စက်တွင်သာ၊ error နှင့် သက်ဆိုင်သည့် packages ကို install လုပ်ပါ။ ဥပမာ—

```bash
sudo apt-get install libegl1 libgl1
```

EGL မရနိုင်သောစက်တွင် Mesa OSMesa libraries ရှိမှ `MUJOCO_GL=osmesa` ဖြင့် စမ်းနိုင်ပါတယ်။ လက်ရှိ workspace တွင် EGL ကိုသာ verify လုပ်ထားပြီး OSMesa ကို မစမ်းသပ်ထားပါ။ Environment တိုင်းအတွက် guaranteed fallback မဟုတ်ပါ။

### ROS plugin error / Python packages ရောနေခြင်း

Venv က `include-system-site-packages=false` ဖြစ်သော်လည်း inherited `PYTHONPATH` မှ ROS packages ဝင်လာနိုင်ပါတယ်။ `env -u PYTHONPATH` နှင့် အထက်ပါ pytest command ကို သုံးပါ။ ROS installation ကို ဖျက်ရန်/ပြင်ရန် မလိုပါ။

### Numba TBB warning

လက်ရှိစက်တွင် TBB version အဟောင်းကြောင့် Numba TBB threading layer disabled warning ပေါ်ပါတယ်။ အစောပိုင်း simulation နှင့် tests ကို မပိတ်ခဲ့ပါ။ Warning ကို အခြား error များနှင့် မရောပါနှင့်။ Process fail လျှင် traceback ကို သီးခြားစစ်ပါ။

### Simulation နှေးနေခြင်း

139k neurons နှင့် sparse connections ~9.5M ကို CPU ပေါ်တွင် update လုပ်ပြီး compound-eye vision ကို render လုပ်နေတာ ဖြစ်ပါတယ်။ Real-time performance အာမခံမထားပါ။ Cache ကို သုံးပြီး `--duration 1.0` မှ စတင်ပါ။ Full connectome build ကို run တိုင်း ပြန်လုပ်ရန် မလိုပါ။

### Fly က လျှောက်ပေမယ့် မလှည့်ခြင်း / spikes မရှိခြင်း

Flat arena တွင် ဘယ်/ညာ brightness ဆင်တူနိုင်ပါတယ်။ လက်ရှိ visual encoder က mean brightness သာဖြစ်သောကြောင့် target tracking မရသေးပါ။ ထို့အပြင် base drive ကြောင့် zero spikes နှင့်လည်း လမ်းလျှောက်နိုင်ပါတယ်။ Output မှ `spike_counts`၊ `motor_rates_hz` နှင့် `drives` ကို အတူကြည့်ပါ။

## ၁၂။ လက်ရှိ မပါသေးသော အရာများ

- Odyssey-3 integration: [OdysseyWorld](./world.py) သည် placeholder သာဖြစ်ပြီး construction မှာ `NotImplementedError` ပေးပါတယ်။ CLI က `FlyGymWorld` ကိုပဲ သုံးပါတယ်။
- Generative/action-conditioned world model backend။
- Visual navigation target နှင့် custom browser dashboard။ Native MuJoCo viewer ကို `--gui`၊ saved MP4 ကို `--video` ဖြင့် သုံးနိုင်ပါပြီ။
- Spatial retinal encoding၊ optic flow သို့မဟုတ် object tracking။
- Cell-type-verified sensory/descending mappings နှင့် biological firing-rate calibration။
- Connectome-based VNC/leg-joint controller။ လက်ရှိ walking controller သည် FlyGym မှ ဖြစ်ပါတယ်။
- Learning/plasticity၊ olfactory-to-brain input နှင့် brain သို့ mechanosensory feedback။

**အနှစ်ချုပ်:** လက်ရှိ system သည် **FlyWire connectivity ပေါ်မှ simplified LIF brain + FlyGym controller/body + MuJoCo world** ကို ချိတ်ထားသော working software prototype ဖြစ်ပါတယ်။ Fly တစ်ကောင်လုံး၏ nervous system ကို biological accuracy ဖြင့် emulation လုပ်ပြီးသားဟု မယူဆသင့်ပါ။

## ရင်းမြစ်များ

- [FlyGym v1.2.1 source နှင့် NeuroMechFly v2 overview](https://github.com/NeLy-EPFL/flygym/tree/v1.2.1)
- [NeuroMechFly documentation](https://neuromechfly.org/)
- [MuJoCo](https://mujoco.org/)
- [FlyWire](https://flywire.ai/)
- [FlyWire connectivity dataset DOI: 10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)
- [Zenodo dataset metadata API](https://zenodo.org/api/records/10676866)

Version၊ file-size/checksum နှင့် saved-run values များကို 2026-10-24 တွင် ဤ workspace မှ စစ်ဆေးထားပါတယ်။ စက်အသစ်ပေါ် clean install အပြည့်နှင့် rendering driver အမျိုးမျိုးကိုတော့ သီးခြား ပြန်မစမ်းသပ်ထားပါ။