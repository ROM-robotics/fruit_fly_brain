# TODO: Wing-based flight model

ဆက်လုပ်ရန်: **2026-10-29**

ရည်ရွယ်ချက်: လက်ရှိ FlyWire SNN + FlyGym walking simulation ကို မပျက်စေဘဲ
MuJoCo wing-based flight backend ကို သီးခြား စမ်းသပ်တည်ဆောက်ရန်။
အောက်ပါ flight features များကို **လက်ရှိ implement မလုပ်ရသေးပါ**။

## လက်ရှိအခြေအနေ

- Brain connectivity: FlyWire v783၊ neurons 139,255 ခု။
- Neural dynamics: NumPy/SciPy LIF network။
- Body/control: FlyGym 1.2.1 / NeuroMechFly v2 + walking controller။
- Host/Docker native MuJoCo GUI၊ headless run နှင့် MP4 export ရှိပြီးသား။
- `docker/run_image` ကို arguments မပေးဘဲ ခေါ်လျှင် walking GUI ကို ဖွင့်သည်။
- လက်ရှိ body ၏ wing meshes မှာ wing joints မပါသဖြင့် walking model ကို
  flag ပြောင်းရုံနှင့် wing-based flight မရနိုင်ပါ။
- လက်ရှိ SNN output pools များသည် GNG-output heuristics ဖြစ်ပြီး
  flight motor neurons အဖြစ် အတည်ပြုထားခြင်း မရှိပါ။

## ၁။ လက်ရှိ walking baseline ကို မှတ်တမ်းတင်ခြင်း

- [ ] လက်ရှိ Git revision နှင့် dependency versions ကို မှတ်ထားရန်။
- [ ] Host tests၊ Docker headless run၊ GUI နှင့် short MP4 export ကို ပြန်စစ်ရန်။
- [ ] Baseline numerical output နှင့် video ကို flight outputs နှင့်
  မရောသော path များတွင် သိမ်းရန်။
- [ ] Walking implementation နှင့် မူလ data/cache များကို မဖျက်ရန်။

**ပြီးမြောက်မှုစံ:** လက်ရှိ walking behavior ကို ပြန်စမ်းနိုင်ပြီး flight
အပြောင်းအလဲများကြောင့် regression ဖြစ်/မဖြစ် နှိုင်းယှဉ်နိုင်ရမည်။

## ၂။ Flybody နှင့် flight controller ရရှိနိုင်မှုကို အရင်အတည်ပြုခြင်း

- [ ] [TuragaLab/flybody](https://github.com/TuragaLab/flybody) ၏ official
  README၊ flight environment examples နှင့် model assets ကို လေ့လာရန်။
- [ ] Wing joints/actuators၊ aerodynamic force model၊ action/observation
  shapes၊ timestep နှင့် model units ကို အတည်ပြုရန်။
- [ ] အသုံးပြုမည့် release သို့မဟုတ် commit ကို ရွေး၍ pin လုပ်ရန်။
- [ ] Python၊ MuJoCo၊ dm_control၊ NumPy compatibility ကို လက်ရှိ environment
  နှင့် နှိုင်းယှဉ်ရန်။
- [ ] Runnable flight controller သို့မဟုတ် pretrained policy/checkpoint
  တကယ်ရရှိမရ၊ ၎င်းကို run ရန် လိုအပ်သော dependencies ကို စစ်ရန်။
- [ ] Code၊ model၊ policy/checkpoint တို့၏ license နှင့် attribution
  လိုအပ်ချက်များကို စစ်ရန်။
- [ ] Controller မရရှိလျှင် controller တည်ဆောက်ခြင်း သို့မဟုတ် policy training
  လိုအပ်မှုကို သီးခြားဆုံးဖြတ်ရန်။ Training အချိန်/compute လိုအပ်ချက်ကို
  မသိဘဲ install လုပ်ရုံနှင့် ပျံမည်ဟု မယူဆရန်။

**ပြီးမြောက်မှုစံ:** အသုံးပြုမည့် model revision၊ flight task၊ controller
နှင့် installation requirements တို့ကို သေချာသတ်မှတ်နိုင်ရမည်။
ဤအဆင့်မပြီးမီ SNN integration မစရန်။

## ၃။ Flight dependencies ကို walking setup နှင့် ခွဲထားခြင်း

- [ ] အဆင့် ၂ တွင် အတည်ပြုထားသော dependencies ကို manifest တွင် သတ်မှတ်ရန်။
- [ ] လက်ရှိ walking requirements ကို မလိုအပ်ဘဲ upgrade/replace မလုပ်ရန်။
  Incompatible dependencies ရှိလျှင် သီးခြား flight Docker image သုံးရန်။
- [ ] Docker ထဲတွင် virtual environment မသုံးဘဲ system Python သို့
  dependencies တိုက်ရိုက် install လုပ်ရန်။
- [ ] Core model run အတွက် မလိုအပ်သော training stack/GPU dependencies ကို
  အလိုအလျောက် မထည့်ရန်။ ရွေးထားသော controller လိုအပ်မှသာ ထည့်ရန်။
- [ ] Dependency validation နှင့် minimal import/model-load smoke test လုပ်ရန်။

**ပြီးမြောက်မှုစံ:** Flight environment ကို reproducibly တည်ဆောက်နိုင်ပြီး
မူလ walking image ကိုလည်း ဆက်သုံးနိုင်ရမည်။

## ၄။ SNN မချိတ်သေးဘဲ standalone flight demo ကို အောင်မြင်အောင်လုပ်ခြင်း

- [ ] Official flight example နှင့် သက်ဆိုင်ရာ controller/policy ကို အရင် run ရန်။
- [ ] Wing motion၊ lift/drag နှင့် body response တို့ တကယ်ချိတ်ဆက်နေကြောင်း စစ်ရန်။
- [ ] Gravity ကို ဖွင့်ထားရန်။ Body position ကို တိုက်ရိုက်ရွှေ့ခြင်း၊
  model ၏ aerodynamic force အစား arbitrary upward force ထည့်ခြင်းကို
  wing-based flight အောင်မြင်မှုအဖြစ် မယူရန်။
- [ ] Model/task ၏ stabilization သို့မဟုတ် assistance forces ရှိလျှင်
  ဘယ်နေရာတွင် သုံးထားကြောင်း မှတ်တမ်းတင်ပြီး အကဲဖြတ်ချိန် ထည့်သွင်းရန်။
- [ ] Random seed ကို ထိန်း၍ reproducible run လုပ်ရန်။
- [ ] Position၊ velocity၊ attitude၊ wing commands နှင့် ground contact ကို
  record လုပ်ရန်။
- [ ] အရင်ဆုံး task က support လုပ်သော stable flight ကို စမ်းရန်။
  Hover၊ ground takeoff၊ landing တို့ကို controller က support လုပ်ကြောင်း
  မစစ်ဘဲ initial demo တစ်ခုထဲမှာ အားလုံးမထည့်ရန်။
- [ ] Short MP4 နှင့် GUI ဖြင့် အမြင်ပိုင်းကိုပါ စစ်ရန်။

**ပထမ milestone:** သတ်မှတ်ထားသော initialization/stabilization ပြီးနောက်
အနည်းဆုံး simulated ၁ စက္ကန့်စာ flight တွင် NaN/Inf၊ physics instability
နှင့် မရည်ရွယ်သော ground contact မရှိရ။ တူညီသော seed ဖြင့် ပြန်စမ်းနိုင်ရ။
Height/attitude tracking tolerance ကို ရွေးထားသော task နှင့် model units
အပေါ်မူတည်၍ ကြိုတင်သတ်မှတ်ပြီး measured error နှင့် စစ်ရန်။

## ၅။ Project ထဲတွင် သီးခြား flight backend ချိတ်ခြင်း

- [ ] လက်ရှိ [world.py](fruitfly_brain/world.py) interface နှင့်
  [closed_loop.py](fruitfly_brain/closed_loop.py) ၏ walking assumptions ကို စစ်ရန်။
- [ ] Flight backend ကို walking backend မှ explicit ရွေးချယ်နိုင်အောင်လုပ်ရန်။
  Default walking behavior ကို မပြောင်းရန်။
- [ ] `reset`၊ `step`၊ position၊ rendering၊ video saving နှင့် resource cleanup
  တို့အတွက် လိုအပ်သလောက် shared contract သတ်မှတ်ရန်။
- [ ] Existing two-element walking drive ကို flight actuator array အဖြစ်
  တိုက်ရိုက် မပို့ရန်။ Flight action bounds/shapes ကို explicit validate လုပ်ရန်။
- [ ] Flight timestep နှင့် LIF timestep ကို စစ်ရန်။ လက်ရှိ
  `brain_every=10` assumption ကို backend အသစ်တွင် မျက်စိမှိတ် မသုံးရန်။
- [ ] Simulation units၊ coordinate frames နှင့် camera tracking scale
  ကွာခြားမှုများကို ကိုင်တွယ်ရန်။
- [ ] GUI window စောပိတ်ခြင်း၊ partial records၊ error reporting နှင့်
  physics/OpenGL resource cleanup ကို ထိန်းသိမ်းရန်။

**ပြီးမြောက်မှုစံ:** Project CLI မှ flight backend ကို ရွေး၍ standalone
controller ဖြင့် run နိုင်ပြီး walking backend လည်း regression မဖြစ်ရ။

## ၆။ FlyWire SNN ကို high-level flight commands နှင့် ချိတ်ခြင်း

- [ ] Stable standalone flight ရပြီးမှ ဤအဆင့်ကို စရန်။
- [ ] Flight backend ရရှိနိုင်သော vision/body-state observations ကို စစ်ပြီး
  SNN sensory encoder အတွက် အသုံးပြုမည့် signals ကို သတ်မှတ်ရန်။
- [ ] ပထမ integration တွင် SNN က high-level commands ပေးပြီး
  အတည်ပြုထားသော flight controller က low-level wing actions ကို ကိုင်တွယ်စေရန်။
- [ ] ရနိုင်သော controller interface အပေါ်မူတည်၍ turn၊ speed၊ height စသည့်
  command mapping ကို ရွေးရန်။ မရှိသော command API ကို မယူဆရန်။
- [ ] Output neuron pools ကို flight အတွက် သက်သေပြနိုင်သော annotations
  ရှိမရှိ စစ်ရန်။ မရှိလျှင် heuristic mapping ဟု ရှင်းလင်းစွာ ဖော်ပြရန်။
- [ ] Rate smoothing၊ scaling၊ saturation နှင့် safe command bounds ထည့်ရန်။
- [ ] Standalone controller၊ SNN-enabled နှင့် SNN-disabled/shuffled controls
  ကို တူညီသော seeds/initial conditions ဖြင့် နှိုင်းယှဉ်ရန်။

**ပြီးမြောက်မှုစံ:** Command tracking နှင့် trajectory metrics ဖြင့် SNN ၏
အကျိုးသက်ရောက်မှုကို ခွဲပြနိုင်ရမည်။ Fly ပျံနေတာတစ်ခုတည်းကို biological
brain control အောင်မြင်မှုဟု မဆိုရ။

## ၇။ Tests၊ GUI/MP4 နှင့် Docker ကို အပြီးချိတ်ခြင်း

- [ ] Action/observation shapes၊ timestep mismatch၊ invalid input၊
  seed reproducibility နှင့် cleanup အတွက် targeted tests ထည့်ရန်။
- [ ] Short real-physics flight smoke test ထည့်ရန်။ လိုအပ်သော optional
  model/policy မရှိလျှင် skip reason ကို ရှင်းလင်းစွာ ပြရန်။
- [ ] Walking regression tests ကို ဆက်အောင်မြင်စေရန်။
- [ ] Headless flight MP4 ကို decode လုပ်၍ frames/FPS ကို စစ်ရန်။
- [ ] Host native MuJoCo GUI နှင့် Docker GUI ကို သီးခြား စမ်းရန်။
- [ ] Docker GUI အတွက် လက်ရှိ GLFW/GLX၊ Xauthority forwarding နှင့်
  non-root UID/GID behavior ကို ထိန်းရန်။
- [ ] ရွေးထားသော model/controller အတွက် software rendering အဆင်ပြေမပြေ၊
  CPU/GPU လိုအပ်ချက်နှင့် memory/runtime ကို တိုင်းတာရန်။
- [ ] GUI၊ numerical records နှင့် MP4 outputs ကို host သို့
  မှန်ကန်သော ownership ဖြင့် persist ဖြစ်ကြောင်း စစ်ရန်။

## ၈။ အသုံးပြုနည်းနှင့် ပထမ flight ရလဒ်ကို မှတ်တမ်းတင်ခြင်း

- [ ] [README.md](README.md) တွင် အတည်ပြုပြီးသော installation၊
  backend selection၊ GUI၊ headless နှင့် MP4 commands ကို ရေးရန်။
- [ ] Model/policy version၊ download sources၊ licenses နှင့် checksums
  ရရှိလျှင် မှတ်တမ်းတင်ရန်။
- [ ] Run တစ်ခုချင်းစီ၏ seed၊ timestep၊ duration၊ controller settings နှင့်
  unit conventions ကို output metadata နှင့်အတူ သိမ်းရန်။
- [ ] ပထမအောင်မြင်သော flight MP4၊ numerical record နှင့် metrics ကို သိမ်းရန်။
- [ ] Hover/takeoff/landing မပါသေးလျှင် မပါသေးကြောင်း တိတိကျကျ ဖော်ပြရန်။

## မနက်ဖန် ဦးစားပေးအစီအစဉ်

1. Walking baseline ကို ပြန်စစ်ရန်။
2. Flybody model/controller/policy ရရှိနိုင်မှုနှင့် compatibility ကို အတည်ပြုရန်။
3. Dependencies ကို ခွဲပြီး standalone wing-based flight demo အရင်ရအောင်လုပ်ရန်။
4. ထို demo အောင်မြင်မှ project backend integration ကို ဆက်လုပ်ရန်။
5. SNN-controlled flight ကို နောက် milestone အဖြစ်သာ ချိတ်ရန်။

**ပထမနေ့အတွက် အဓိကရလဒ်:** အထောက်အထားပါ standalone wing-based flight
demo တစ်ခု၊ သို့မဟုတ် မရနိုင်သေးလျှင် controller/dependency blockers နှင့်
နောက်လုပ်ရမည့်အဆင့်ကို တိတိကျကျ သတ်မှတ်ထားနိုင်ခြင်း။
