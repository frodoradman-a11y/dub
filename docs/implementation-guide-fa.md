# راهنمای کامل پیاده‌سازی DubFlow

> این سند پیاده‌سازی طرح [`dubflow-plan-b-fa.md`](dubflow-plan-b-fa.md) را توضیح می‌دهد: ساختار کد، فرمت داده‌ها، اجرای مرحله‌به‌مرحله روی سرور GPU، پیکربندی‌ها، منابع لازم، تست‌ها، و محدودیت‌های شناخته‌شده.

---

## ۱. خلاصه

| | DubFlow-M (کیفیت) | DubFlow-S (سرعت) |
|---|---|---|
| بک‌بون | DiT با ۲۴ لایه، ‏d=1280، skipهای U-ViT، attention محلی و سراسری | U-Net چندنرخی (۲۵ → ۱۲٫۵ → ۶٫۲۵ → ۱۲٫۵ → ۲۵ هرتز)، ‏d=768 |
| پارامترها (بدون VAE) | **۸۱۵M** (بک‌بون ۷۵۱M، متن ۴۰M، تیمبر ۱۰M) | **۱۹۸M** (بک‌بون ۱۷۰M، متن ۱۹M) |
| DubVAE (مشترک) | ۱۶۸M (انکودر ۵۱M و دیکودر ۱۱۷M) | همان |
| پیکربندی | `configs/dubflow_m.yaml` | `configs/dubflow_s.yaml` |

اعداد بالا با `dubflow-params` از خود کد محاسبه شده‌اند.

**ایدهٔ اصلی:** مدل یک صحنه (تا ۶۰ ثانیه) را یکجا تولید می‌کند. هویت و لحن **هر لحظه** از **همان لحظهٔ صدای اصلی** گرفته می‌شود (TSVB). هیچ برچسب گوینده و هیچ دیاریزیشنی وجود ندارد. متن ترجمه به‌صورت «قطعه‌های زمانی» می‌آید و توکن‌های هر قطعه یکنواخت روی بازهٔ زمانی همان قطعه پخش می‌شوند (average upsampling لنگرشده).

---

## ۲. نگاشت طرح به کد

| مفهوم در طرح | فایل |
|---|---|
| DubVAE، یعنی ۲۴ کیلوهرتز ⇄ latent ‏۲۵ هرتز × ۶۴، دیکودر به سبک BigVGAN، و هد معنایی RedAE | `dubflow/vae/model.py`، `dubflow/vae/discriminators.py` |
| انکودر متن نویسه‌ای و LID دوسطحی (FiLM روی embedding، و AdaLN) | `dubflow/model/conditioning.py` (`TextEncoder`)، `dubflow/model/dubflow.py` |
| average upsampling لنگرشده به زمان | `dubflow/data/tracks.py` (`build_frame_tok`)، `conditioning.anchored_upsample` |
| ردّهای هم‌زمان: هویت جزیره‌ای، هویت کوتاه‌پنجره، لحن درشت، احساس و فعالیت | `dubflow/data/tracks.py`، `conditioning.TrackProjector` |
| پنجرهٔ تیمبر ±۱ ثانیه (cross-attention باندی با null-key) | `conditioning.gather_windows`، `TimbreWindowAttention` |
| offset-window در پیش‌آموزش | `tracks.window_centers_offset` |
| صحنه‌های مصنوعی hard-negative (صداهای نزدیک درون یک قطعه) | `dubflow/data/datasets.py` (`PretrainDataset`) |
| بک‌بون M (DiT و U-ViT) و بک‌بون S (U-Net چندنرخی) | `dubflow/model/backbones.py` |
| CFG چندشرطی (all، text، timbre، tracks) | `DubFlow.sample` |
| REPA با w2v-BERT 2.0 (به‌صورت آنلاین) | `DubFlow.forward`، `features.SSLTeacher` |
| Flow-GRPO با پنجرهٔ SDE و ادغام قیدی پاداش‌ها | `dubflow/rl/flow_grpo.py`، `dubflow/train/train_grpo.py` |
| تقطیر چندگام به چندگام کمتر، دانش‌آموز آگاه از گام (M→M یا M→S) | `dubflow/train/distill.py` |
| معیارها (SIM جزیره‌ای، SSP، boundary، CER/WER، SLC، Pause-IoU) | `dubflow/eval/metrics.py` |

---

## ۳. نصب

```bash
git clone <repo> && cd dub
bash scripts/00_setup.sh          # venv، نصب وابستگی‌ها، اجرای تست‌ها
```

- **نیازمندی‌ها:** پایتون ۳٫۱۰ یا بالاتر، PyTorch ۲٫۳ یا بالاتر با CUDA.
- **وابستگی‌های اختیاری** (`.[features]`): ‏`transformers` (برای WavLM-SV، w2v-BERT 2.0 و Whisper)، `funasr` (برای emotion2vec)، `speechbrain` (برای ECAPA، اختیاری).
- **هم‌ترازی کلمه‌ای فارسی:** `pip install uroman`.
- **دانلود مدل‌ها:** در اولین اجرا خودکار انجام می‌شود (HuggingFace و ModelScope). اگر سرور اینترنت ندارد، پوشه‌های `~/.cache/huggingface` و `~/.cache/modelscope` را از یک ماشین دیگر کپی کنید.

---

## ۴. فرمت داده‌ها

### ۴.۱ مانیفست پیش‌آموزش (`data/pretrain_manifest.jsonl`)، یک خط برای هر گفتار

```json
{"id": "en_000123", "audio": "/data/en/000123.wav", "lang": "en", "speaker": "spk_42",
 "text": "hello there", "words": [{"w": "hello", "s": 0.12, "e": 0.48}, {"w": "there", "s": 0.55, "e": 0.90}]}
```

- `lang` یکی از `en`، `fa`، `zh` یا `other` است. **متن چینی باید pinyin با شمارهٔ لحن باشد** (مثلاً با `pypinyin`).
- `words` اختیاری است، ولی **به‌شدت توصیه می‌شود**، چون مرز قطعه‌ها از آن ساخته می‌شود. اگر ندارید: `python -m dubflow.preprocess.align --manifest in.jsonl --out out.jsonl` (MMS_FA، با uroman برای فارسی).
- `speaker` برای انتخاب شریک hard-negative در صحنه‌های مصنوعی استفاده می‌شود. اگر نباشد، `id` به جای آن استفاده می‌شود.

### ۴.۲ مانیفست VAE (`data/vae_manifest.jsonl`)

فقط `{"id", "audio"}` لازم است. هر گفتاری مناسب است. مقداری صدای غیرگفتاری هم اضافه کنید (خنده، نفس، فریاد).

### ۴.۳ فیلم‌های دوبله (`data/films.tsv`)

هر خط: `film_id <TAB> vocals_en.wav <TAB> vocals_fa_vc.wav <TAB> script_fa.json|srt`

- هر دو فایل صوتی **وکالِ جداشده** (بدون موسیقی و افکت) و **روی یک timeline** هستند.
- تراک فارسی همان **نسخهٔ VC‌شده به صدای بازیگر اصلی** است.
- اسکریپت (JSON یا SRT):
  ```json
  {"chunks": [{"t0": 12.40, "t1": 14.10, "text": "متن فارسی با کسرهٔ اضافهِ لازم", "ok": true}]}
  ```
  `ok: false` یعنی VC آن خط خراب است. آن خط فقط context می‌شود و loss نمی‌گیرد.

### ۴.۴ اسکریپت استنتاج (خروجی مترجم بیرونی)

```json
{"lang": "fa",
 "chunks": [{"t0": 12.40, "t1": 14.10, "text": "<emph>واقعاً</emph> رفتی؟", "stream": 0},
            {"t0": 14.10, "t1": 15.30, "text": "آره <laugh>", "stream": 0}]}
```

- **واحد، قطعهٔ زمانی است، نه خط زیرنویس.** قطعه‌ها را روی مکث‌های متن اصلی بشکنید.
- **تگ‌های پشتیبانی‌شده:** `<emph>…</emph>`، `<whisper>…</whisper>`، `<shout>…</shout>`، `<laugh>`، `<breath>`، `<sigh>`، `<cry>`، `<cough>`، `<gasp>`، `<scream>`، `<pause>`، `<hum>`.
- **اعداد را با حروف بنویسید.** کسرهٔ اضافه (ِ) و مصوت‌های مبهم را بگذارید.
- `stream: 1` فقط برای گفتار هم‌پوشان است (بخش ۱۱ را ببینید).

---

## ۵. اجرای مرحله‌به‌مرحله

پیش‌فرض‌ها در `scripts/env.sh` تنظیم می‌شوند (تعداد GPU، پیکربندی، backendها). برای مدل S:

```bash
export CFG=configs/dubflow_s.yaml RUN=runs/s
```

| # | دستور | خروجی | توضیح |
|---|---|---|---|
| D0 | `bash scripts/10_train_vae.sh` | `runs/vae/last.pt` | ۲۰ هزار گام اول فقط بازسازی است، بعد GAN فعال می‌شود |
| — | `bash scripts/20_extract_pretrain.sh` | `feats/pretrain/*.pt`، ‏`index.pt`، ‏`latent_stats.pt` | به ازای هر GPU یک shard. با `--skip-existing` قابل ادامه است |
| D1 | `bash scripts/30_train_d1.sh` | `runs/m/d1/last.pt` | REPA تا گام ۲۰۰ هزار فعال است |
| — | `bash scripts/40_build_dub.sh` | `data/dub/pairs.jsonl`، ‏`feats/dub_src`، ‏`feats/dub_tgt` | ردّ کوتاه‌پنجرهٔ واقعی فقط برای سمت مبدأ ساخته می‌شود |
| D2 | `bash scripts/50_train_d2.sh` | `runs/m/d2/last.pt` | آمار نرمال‌سازی latent از D1 به ارث می‌رسد |
| D3 | `bash scripts/60_train_d3_grpo.sh` | `runs/m/d3/last.pt` | برای پاداش CER: ‏`grpo.asr_model=<whisper فارسی>` |
| D4 | `bash scripts/70_distill.sh` | `runs/m/d4/last.pt` | M→S: ‏`STUDENT_CFG=configs/dubflow_s.yaml` |
| دوبله | `bash scripts/80_dub.sh vocals_en.wav script_fa.json out.wav me.wav` | wav ‏۲۴ کیلوهرتز | با student، ۴ گام و بدون CFG |

**نکات:**
- **ادامهٔ آموزش (resume):** کافی است همان دستور را دوباره اجرا کنید. `last.pt` در `train.out_dir` خودکار بارگذاری می‌شود.
- **override تنظیمات:** هر کلیدی از پیکربندی را می‌شود از خط فرمان عوض کرد، مثلاً `train.lr=5e-5 data.max_frames=1000`.
- **نمونه‌های صوتی در طول آموزش:** با `--vae` در `train_dubflow`، در هر ذخیره یک فایل در `samples/` نوشته می‌شود.
- **لاگ‌ها:** در `train_log.txt` و TensorBoard (`tb/`).

---

## ۶. پیکربندی‌های مهم

| کلید | معنا |
|---|---|
| `model.tracks.*` | ابعاد ردّها. **باید با backend استخراج یکی باشد** (WavLM-SV ‏۵۱۲، ECAPA ‏۱۹۲، emotion ‏۶۴). اگر نباشد، آموزش با پیام خطای واضح متوقف می‌شود |
| `model.timbre.window_half` | نیم‌پهنای پنجرهٔ تیمبر (۲۵ = ±۱ ثانیه) |
| `model.cfg_dropout` | احتمال حذف هر شرط برای CFG |
| `data.lang_weights` | upsampling زبان‌ها (پیش‌فرض: فارسی ×۶) |
| `data.scene_p`، `hard_negative_p`، `no_gap_p`، `cross_speaker_chunk_p` | ساخت صحنه‌های مصنوعی و حالت‌های سخت |
| `data.window_min_offset_sec` | حداقل فاصلهٔ offset-window (۳ ثانیه) |
| `sample.guidance` | ضرایب CFG در استنتاج teacher (پیش‌فرض: all برابر ۱٫۵، timbre برابر ۰٫۵) |
| `grpo.*` | اندازهٔ گروه، پنجرهٔ SDE، وزن پاداش‌ها و gateها |
| `distill.*` | تعداد گام student، زیرگام‌های teacher، و احتمال تمرین یک‌گامی |

---

## ۷. منابع و زمان (تخمین)

| | M | S |
|---|---|---|
| VRAM استنتاج | حدود ۴ تا ۶ GB (به‌علاوهٔ مدل‌های ویژگی، حدود ۲ تا ۳ GB) | حدود ۲ تا ۳ GB |
| آموزش | کارت ۴۰ تا ۸۰ GB پیشنهاد می‌شود. با `activation_checkpointing` روی ۲۴ GB هم ممکن است، ولی batch کوچک | روی ۲۴ GB (3090 و 4090) راحت |
| D1 روی ۸ کارت H100 | حدود ۱ تا ۲ هفته | چند روز |

**اندازه‌گیری روی CPU** (۴ هستهٔ CPU، صحنهٔ ۳۰ ثانیه‌ای، صرفاً برای اثبات اجرا):
- یک گام کامل آموزش S (forward و backward): ۹٫۳ ثانیه، با پیک حافظهٔ ۲٫۷ GB
- یک گام ODE مدل M: ۶٫۸ ثانیه، با پیک حافظهٔ ۵٫۵ GB

روی GPU این زمان‌ها چند ده برابر کمتر است.

**فضای دیسک feature cache:** برای هر ساعت گفتار حدود ۱۲ MB برای latent، به‌علاوهٔ ردّها (fp16). برای ۵۰ هزار ساعت، حدود **۶۰۰ تا ۸۰۰ GB**. REPA آنلاین حساب می‌شود و چیزی ذخیره نمی‌کند.

---

## ۸. تست‌ها و آنچه واقعاً آزموده شده

```bash
python -m pytest tests -q        # ۱۶ تست، حدود ۳۵ ثانیه روی CPU
```

| آزموده‌شده | روش |
|---|---|
| توکنایزر فارسی (نرمال‌سازی، کسرهٔ اضافه، تگ‌ها) | تست واحد |
| upsampling لنگرشده، offset-window، پنجرهٔ تیمبر، شبیه‌سازی پنجرهٔ لغزان | تست واحد |
| **عدم نشت padding** در هر دو بک‌بون | تست واحد |
| **قابلیت یادگیری** (overfit روی یک batch) در هر دو بک‌بون | تست واحد |
| GRPO: نسبت برابر ۱ و KL برابر ۰ بدون به‌روزرسانی، و جریان گرادیان | تست واحد |
| ادغام قیدی پاداش‌ها (gate، و گروهی که هیچ عضوش رد نمی‌شود) | تست واحد |
| بودجهٔ پارامتر M و S | تست واحد (روی meta device) |
| **کل خط لوله:** VAE، extract، index، D1 (DiT و U-Net)، ساخت جفت دوبله، D2، GRPO، تقطیر DiT→U-Net، دوبلهٔ CLI (teacher و student) | تست end-to-end روی دادهٔ مصنوعی |
| **اجرای چندپردازه‌ای (DDP)** همهٔ trainerها | `torchrun --nproc_per_node=2` (gloo روی CPU) |
| **backendهای واقعی:** WavLM-SV (۵۱۲ بُعدی، تفکیک صداها)، w2v-BERT 2.0، emotion2vec، و هم‌ترازی MMS_FA | اجرا با وزن‌های واقعی |
| **مدل‌های با اندازهٔ واقعی** S و M روی صحنهٔ ۳۰ ثانیه‌ای | اجرا روی CPU |

**آزموده‌نشده (در این محیط GPU نبود):**
- آموزش واقعی تا همگرایی
- bf16 و NCCL روی GPU
- کیفیت صوتی نهایی

**قبل از آموزش اصلی:** آزمایش‌های G1 تا G7 طرح را روی مقیاس کوچک اجرا کنید.

---

## ۹. عیب‌یابی

| نشانه | علت و راه‌حل |
|---|---|
| `model.tracks.X=... but feature cache has ...` | backend استخراج با پیکربندی نمی‌خواند. یا `model.tracks` را اصلاح کنید یا backend را |
| OOM در D1 | `train.batch_size` یا `data.max_frames` را کم کنید، و `train.activation_checkpointing=true` بگذارید |
| loss ‏REPA صفر یا ثابت | `repa_teacher.until_step` تمام شده، یا `wav` در batch نیست (فیلد `audio` مانیفست را چک کنید) |
| صدای دوبله صدای اشتباه است | ردّها در اجرای `extract` برای سمت مبدأ با `--short-window` ساخته شده باشند |
| در GRPO، مقدار reward همیشه منفی است | gateها برای مدل فعلی سخت‌اند. `grpo.gates.boundary` را موقتاً بیشتر کنید (هنوز سیگنال یادگیری دارد) |
| گفتار از قطعه بیرون می‌زند | زمان‌بندی مترجم را چک کنید (`t0` و `t1`). در استنتاج `--max-scene-sec` را کم نکنید |

---

## ۱۰. تفاوت‌های پیاده‌سازی با طرح (صادقانه)

1. **بک‌بون S یک «U-Net چندنرخی الهام‌گرفته از Zipformer» است، نه Zipformer کامل** (BiasNorm، SwooshR، NLA و ScaledAdam پیاده نشده‌اند). bypassهای یادگرفتنی و پردازش چندنرخی حفظ شده‌اند. **بهینه‌ساز هر دو مدل AdamW است.** Muon و ScaledAdam برای آزمایش G4 گذاشته شده‌اند.
2. **discriminator‌های VAE** MPD و STFT چندرزولوشنی هستند (به جای CQT-D).
3. **تخمین F0 داخلی** مبتنی بر autocorrelation است. برای ردّ لحن درشت (۵ هرتز) کافی است. اگر لازم شد، RMVPE یا PENN را در `utils/audio.estimate_f0` جایگزین کنید.
4. **VAD داخلی** انرژی‌محور است. برای صدای جداشده کافی است. Silero را می‌شود در `utils/audio.energy_vad` جایگزین کرد.
5. **هم‌پوشانی گفتار:** مدل دو جریان متن دارد (`stream`)، ولی **صحنه‌های مصنوعی پیش‌آموزش overlap نمی‌سازند**، چون latent مخلوط با جمع latentها برابر نیست. یادگیری overlap فقط از دادهٔ واقعی دوبله ممکن است. پشتیبانی کامل overlap (استخراج گوینده با سرنخ متن و ردّ دوکاناله) در این نسخه **پیاده نشده** است.
6. **مسیر جایگزین mel به‌علاوهٔ vocoder** (fallback در صورت شکست G1) پیاده نشده است. فقط DubVAE.

---

## ۱۱. مسیرهای توسعهٔ بعدی

- پشتیبانی کامل overlap (استخراج گوینده با سرنخ متن و دو ردّ هویت)
- Muon، ScaledAdam، و MeanFlow از صفر (آزمایش‌های G4)
- Reward Model ترجیح انسانی برای GRPO
- streaming با پنجره‌های صحنه و دیکودر causal
