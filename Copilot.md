## Q: မကြာခင်က fruit fly brain ကို Odssey-3 မှာ run ထားတာတွေ့လို့ အဲ့တာကို လိုက်လုပ်ချင်တာပါ။

## အဖြေ

> စစ်ဆေးမှတ်တမ်း (2026-10-24): Odyssey အတည်ပြုချက်များ၊ FlyWire / FlyGym အချက်အလက်များကို primary sources (odyssey.systems, flywire.ai, Glitchwire, Nature 2024, FlyGym docs) ဖြင့် စစ်ဆေးထားပါသည်။

### ၁။ Odyssey-3 Fruit Fly demo

**အတည်ပြုပြီး**
- Odyssey (@odysseyml) က 2026-10-17 X post တွင် fruit fly brain ကို Odyssey-3 ထဲ ထည့်ပြီး ကိုယ်တိုင် AI-generated world ထဲ ပျံသွားတဲ့ demo ကို ပြသခဲ့ပါတယ်။
- Odyssey-3 သည် research preview (2026-10-08 စတင်) ဖြစ်ပါတယ်။ Prompt ဖြင့် environment ဖန်တီးပြီး agent action အပေါ် မူတည်၍ real-time ခန့်မှန်းပါတယ်။
- Architecture: **autoregressive diffusion transformer** ဖြစ်ပါတယ် (visual transformer မဟုတ်ပါ)။ Action-conditioned ဖြစ်ပြီး real-time အတွက် distilled few-step variant ရှိပါတယ်။
- Resolution: Odyssey-3 832×480၊ Pro 1280×720။

**အတည်မပြုနိုင်သေးတာ (Odyssey က ထုတ်ပြန်မထားပါ)**
- ဘယ် connectome ကို သုံးသလဲ (FlyWire သို့မဟုတ် MaleCNS)။
- Sensory input ကို brain ထဲ ဘယ်လို ထည့်သလဲ၊ motor output ကို action အဖြစ် ဘယ်လို ပြောင်းသလဲ။ (Pitch/Roll/Yaw/Speed စသည်တို့မှာ ခန့်မှန်းချက်သာ)
- 14B parameters (Glitchwire သတင်းတွင်သာ ပါပါသည်)။
- `step(action)` ကဲ့သို့ API interface။ Odyssey-3 အတွက် public API/SDK docs မတွေ့ပါ။ Access က research preview ဖြစ်ပြီး ကုမ္ပဏီကို ဆက်သွယ်ရန် လိုပါသည်။
- Paper သို့မဟုတ် technical description မထုတ်ပြန်သေးပါ။

### ၂။ FlyWire connectome
- Adult female *Drosophila melanogaster* ၏ whole-brain connectome ဖြစ်ပါတယ်။ နျူရွန် **139,255** ခု၊ chemical synapse **~50 သန်း** (အရင်းအမြစ်အချို့က 54.5 သန်းဟု ဖော်ပြ)။ Dorkenwald et al., *Nature* 2024။
- Data နှင့် tools: [flywire.ai](https://flywire.ai) (Princeton ၏ Murthy/Seung labs ဦးဆောင်သော FlyWire Consortium)။

### ၃။ Open-source အစိတ်အပိုင်းများ
- **FlyGym / NeuroMechFly v2** (EPFL NeLy lab): MuJoCo-based fly digital twin။ Python package (`pip install flygym`)။ Vision (ommatidia), olfaction, mechanosensory feedback နှင့် CNS control hooks ပါဝင်။ PyTorch မဟုတ်ပါ။
- **Brian2 / NEST**: Spiking neural network simulators။
- **Odyssey-3**: access လိုအပ်ပါသည်။ Code တွင် adapter interface ဖြင့်သာ ချိတ်ထားပြီး access ရလာမှ အစစ်အမှန် ဖြည့်ရမည်။
- Alternative world models (ဥပမာ Diamond, Oasis) ကို open-source ဟု ဖော်ပြထားသော်လည်း ဤအဖြေအတွက် မစစ်ဆေးရသေးပါ။ OpenAI Sora သည် closed-source ဖြစ်ပါတယ်။

### ၄။ Closed-loop pipeline (ကျွန်တော့် ဒီဇိုင်း — Odyssey ၏ နည်းလမ်း မဟုတ်ပါ)

```
[ Odyssey-3 World Model ] ──(RGB frame)──> [ Retina / Vision encoder ]
          ▲                                             │
          │ action                                      ▼
[ Action decoder ] <── [ Motor neuron groups ] <── [ Connectome SNN (FlyWire) ]
```

1. **Frame ရယူ:** Odyssey-3 adapter မှ RGB frame ရယူ။
2. **Retinal encoding:** Frame ကို ommatidia-like sampling / contrast encoding ဖြင့် sensory input အဖြစ် ပြောင်း။
3. **SNN forward pass:** Sensory currents ကို FlyWire wiring ပေါ်တွင် LIF network ဖြင့် run ၍ firing rate ထုတ်။
4. **Motor decoding:** Motor neuron groups ၏ rate မှ action (forward speed, turn) ကို တွက်။
5. **Feedback:** Action ကို world model သို့ ပြန်ပို့ပြီး frame အသစ် ရယူ။ Loop ကို ဆက်လုပ်။

### ၅။ နောက်တစ်ဆင့် (Code)
- FlyWire connectome data ကို download လုပ်ပြီး LIF SNN ကို ဆောက်ခြင်း။
- FlyGym နှင့် ချိတ်ဆက်ခြင်း (body + vision)။
- Odyssey-3 adapter (access ရှိမှ ဖြည့်ရန် stub ဖြင့် စတင်)။

