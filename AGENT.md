# AGENT.md — Master Context & Task Tracking

**This file is the source of truth.** Read it fully before touching anything. It must be able to
reconstruct the whole project after a context loss.

**Rules:**
1. Every meaningful change → one line in *Task Log* + one line in *Decision Log*. Keep *Latest* current.
2. **Always update every MD** in this repo (README, GUIDE, BUILD_YOUR_OWN, CONCLUSION,
   ERROR_PLAYBOOK) after any significant change — not just AGENT.md.
3. The notebook is **generated**, never hand-edited: edit the build script → re-run → `ast`-validate
   all cells → `kaggle kernels push -p .` → commit/push repo + all MDs.
4. Never commit raw tokens to this public repo (HF token / GitHub PAT live outside git or assembled
   at runtime).

**Date:** Oct 8 2026
**Owner:** adityahalde8777
**Kaggle kernel:** `adityahalde8777/minimax-h3-comfy-2xt4-generator` (the ONLY kernel; old lanes deleted)
**Latest:** **v21 PUSHED — v20 run root-caused & fixed (music save-node DynamicCombo serialization).**
v20 run result: the **numpy heal worked EXACTLY as designed** — install cell printed `task deps
health: OK` (subprocess) yet `in-kernel task deps import FAIL: ImportError cannot import name
'_slice' from 'numpy._core.umath'` → heal `pip force-reinstall --no-deps numpy==2.1.3`
(kernel-loaded) → both probes re-verified `OK` → **`TASKS_HEALTHY = True`**. All three smokes
validated (`node_errors: {}`); the **music pipeline executed for real** (AR sampling 57%
72/126 @ 1.17 s/it, tiled decode ran, lazy `ComfySwitchNode` correctly skipped `VAEDecodeAudio`)
— but the run ended **ERROR at the very last step**:
`SaveAudioAdvanced.execute() missing 1 required positional argument: 'format'`.
**Root cause (proven against a local CPU ComfyUI install, torch 2.7.1):**
SaveAudioAdvanced.`format` is an **`IO.DynamicCombo`**, and the ComfyUI *API prompt* must carry
the option **key as a plain string** (`"flac"`); the executor then matches that option, injects
its sub-inputs (mp3/opus → `quality`), and **re-nests** the value into `{"format": ...}` at call
time. v20 sent the dict `{"format":"flac"}` (the UI/widget form) → the option-key match fails
**silently** → the input vanishes from the node's kwargs → `missing positional 'format'`. Proved
locally: `io.get_finalized_class_inputs` + `io.build_nested_inputs` drop the dict form (repro of
the exact v20 crash signature) and nest the string form into exactly what `execute()` reads;
the fixed full music prompt submits to the local server with **no node_errors for node 35**.
**v21 fix**: `_build_music` writes `format` as the plain option-key string (mp3/opus also set the
dotted `format.quality` sub-input), and `_flatten` gained dotted-key support (emit dynamic
sub-inputs when their parent input is present). Rebuilt (13 code cells `ast`-OK), committed,
**PUSHED as kernel version 21** — verify: `TASKS_HEALTHY=True`, `SMOKE PASSED:
['image','video','music']` with a real `h3_music_*.flac` in output, plus healthy
bg_remove/extract through the tunnel.
Prior verified baselines: **v18** (first green run) → **v19** (numpy root-cause proven) → v20
(heal + music run to the last op).
Prior verified baseline: **v18** (COMPLETED, first fully green run; task bug chain root-caused as
self-inflicted pop+reimport → v19's non-destructive verify) → **v19** (above).

---

## Mission (expanded scope — user directives, Oct 7)

The product is an **all-in-one generation hub** on one public Base URL, not just an H3 video demo:

| Modality | Models (candidates) | Status |
|---|---|---|
| **Video generation** | MiniMax H3 (FL2VA/Ref2VA, turbo/dense), H3-Max-style variants | ✅ H3 lane in production |
| **Image generation** | Flux family (Comfy-Org repos), community Fluxes | ✅ v18 smoke proven on Kaggle (PNG + MP4, dual-GPU, **run COMPLETED**) |
| **Image/audio use-case tasks** | bg remover, element extractor, upscaler, frame/GIF/audio tools | 🟡 upscale + video_frames green on Kaggle (v18/v19); bg_remove/extract blocked every boot by the **in-place numpy swap** (subprocess pip upgrades numpy after the kernel pre-loaded it → kernel-only `_slice` ImportError). **v20's heal fired on the in-kernel probe exactly as designed → `TASKS_HEALTHY=True`**, but the v20 run ERRORed at the music save step *before* the pub task smoke, so bg_remove/extract are **still unverified through the tunnel — expected to go green in the v21 run** (same heal) |
| **Music generation** | `audio_minimax_music_3` Comfy template (MiniMax Music 3, **core** ComfyUI nodes), LTx community | 🟡 **v20 ran the FULL pipeline** (30-step AR sampling, tiled decode, lazy switch) but ERRORed at the save step: `SaveAudioAdvanced` missing `format` → **root-caused + proven locally** (DynamicCombo option-key must be a plain string in the API prompt) → **v21 fix PUSHED**, awaiting Kaggle verification |
| **Sound / soundtrack** | H3 native audio track (already joint audio+video), music models | ⬜ Phase C |
| **Transcription (ASR)** | to research (Whisper-class on Comfy / community) | ⬜ Phase C research (v21) |
| **Text generation** | ❌ explicitly NOT needed | — |

Requirements distilled from user messages:
- **Flexible multi-model system**: list *all* kinds of models + advanced settings; selecting a
  model downloads it **at that moment**; switching models **unloads/deletes the previous** so
  space and GPU stay clean and fast; **new models auto-discovered and auto-listed** (live HF
  query each boot — never a static list).
- **Kaggle Dataset auto-cache**: first download → saved to Kaggle Dataset → next boot auto-detected
  under `/kaggle/input`, mirrored into model dirs, no re-download. **200 GB dataset cap**: if the
  dataset is full, the system must detect it, **skip the upload automatically and continue** — never fail.
- **HF token** (user-provided, stored in Kaggle Secret `HF_TOKEN` with embedded runtime fallback)
  for fast/rate-limit-free downloads.
- **Smoke strategy**: one **low-quality smoke per modality/category** to verify THE SYSTEM works —
  explicitly *not* every model (that would be worst-case time). First one: H3 **4-step turbo LoRA @
  480p** (megapixels 0.4 → 864×480) — fast + verifies dual GPU.
- **Enterprise API**: hybrid dynamic management, world-class JSON, full documentation for every
  endpoint/helper (served from the Base URL + a repo MD). Settings surface: model selection,
  resolution/aspect, duration, steps, turbo on/off, LoRA, seed, prompt, first/last frame images,
  download/upload of custom models, load/unload. (H3 has **no negative prompt** — CFG-distilled.)
- Every run's public Base URL + endpoints printed only after its smoke passes.

---

## Key paths & build pipeline

| Path | What |
|---|---|
| `/home/limitlessjourney829/kaggle/minimax-h3-api/build_full.py` | **Notebook generator — NOW IN THE REPO (Phase E done)**; edit → `python3 build_full.py` writes the `.ipynb`; cells embedded verbatim as raw `r'''…'''` blocks; `H3_NB_OUT` overrides the output path |
| `/home/limitlessjourney829/kaggle/minimax-h3-api/minimax_h3_full_comfy.ipynb` | generated notebook (pushed with `kaggle kernels push -p .`; **never hand-edited**) |
| `/home/limitlessjourney829/kaggle/minimax-h3-api/kernel-metadata.json` | kernel metadata (private, GPU, internet, T4) |
| `/home/limitlessjourney829/kaggle/minimax-h3-api/dryrun.py` (⬜ to rebuild) | local CPU dry-run harness (was `/tmp/opencode/dryrun.py`, 59/59; **lost with the host restart — MUST be reconstructed**) |
| `/tmp/opencode/h3_t2v.json` (⚠ wiped with restart — re-fetch on need) | official template `video_minimax_h3_t2v.json` (local copy, converter dry-runs) |
| `/tmp/opencode/ComfyUI/` | local ComfyUI clone (CPU dry-runs; needs pip CPU install) |
| `/tmp/opencode/Comfy-H3-MultiStream/` | node-pack clone (docs, perf, split.py log lines) |
| Repo | https://github.com/adittaya/kaggle-model-notebook-implementation-guide (public) |

Notebook cell order (v21, 14 cells: title + 13 code): title → secrets → preflight → install →
registry → settings → assets → convert → start → api → smoke → wait → pub → sync. The convert
cell holds the shared `build_prompt(cfg)` (branches: video / image / music) + `_flatten`.

---

## Template mechanics (verified by source/JSON inspection — do not re-research)

**Workflow:** official `video_minimax_h3_t2v.json`, single subgraph instance (node 140) + top-level
`SaveVideo`, `ResolutionSelector` (node 115), 3× `MarkdownNote` (UI-only, skipped).

- **Turbo wiring:** subgraph input `value` (label `turbo_mode`) → `PrimitiveBoolean #139` → drives
  **BOTH** `ComfySwitchNode #135` (LoRA branch: `on_true = LoraLoaderModelOnly #134`, `on_false =
  UNETLoader direct`) and `#136` (steps: `on_true = PrimitiveInt #138` ← `turbo_steps` subgraph
  input, `on_false = PrimitiveInt #137` = 20 dense) → `BasicScheduler #124.steps`.
  **One boolean flips LoRA + step count.** `strength_model_1` = turbo LoRA strength (default 1).
- **Instance named widgets** (`widgets_values_named`, both key styles exist): `prompt`, `width`,
  `height`, `value_1` (duration, s), `noise_seed`, `unet_name`, `clip_name`, `vae_name`,
  `vae_name_1` (audio VAE), `value` (turbo bool), `lora_name`, `strength_model_1`, `value_2`
  (turbo_steps; template default 8).
- **Resolution:** `ResolutionSelector{aspect_ratio, megapixels, multiple=32}`. Template default
  **megapixels 0.4 → 864×480 (16:9)**; quality preset 0.98 → 1344×768. Table: 0.2→608×352,
  0.3→736×416, 0.4→864×480, 0.5→960×544, 0.6→1056×608, 0.7→1152×640, 0.8→1216×672, 0.9→1280×736,
  0.98→1344×768 (multiple 32; other aspects available).
- **Duration:** seconds → frames snaps to `17k+5` @ 24 fps (124 frames = 5 s; trained range
  ~124–362, longer untested; node accepts length 5–3600).
- **No negative prompt anywhere**: H3 is CFG-distilled (CFG 1); `MiniMaxH3ImageToVideo` inputs =
  `clip, vae, first_frame, last_frame, prompt, width, height, length`. Do NOT expose
  `negative_prompt` in the H3 API schema.
- **I2V ready:** subgraph exposes `first_frame`/`last_frame` IMAGE inputs (unlinked in T2V
  template); FL2VA = 0/1/2 images (t2v, first-frame, last-frame, first+last).
- **Embeddings:** put `embedding:minimaxh3_<name>` in the prompt (10 style embeddings in repo).
- **Template default models:** `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (DiT 21 GB),
  `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (TE 15.7 GB), `minimax_h3_video_vae_int8_convrot.safetensors`
  (2.8 GB), `minimax_h3_audio_vae_fp32.safetensors` (0.6 GB), turbo LoRA (1.96 GB). ≈41 GB total.
- **Official API-format templates exist** (no UI conversion needed for future lanes):
  `api_minimax_h3_t2v.json`, `api_minimax_h3_r2v.json`, `api_minimax_h3_max_{t2v,i2v,r2v,turbo_*}.json`,
  `audio_minimax_music_3.json` in `Comfy-Org/workflow_templates`.
- **ComfyUI native H3 nodes** (`comfy_extras/nodes_minimax_h3.py`): `MiniMaxH3ImageToVideo`,
  `MiniMaxH3AddGuide`, `MiniMaxH3ReferenceToVideo` (Ref2VA!), `MiniMaxH3SigmaShift`,
  `MiniMaxH3FunControlNetApply`, `EmptyMiniMaxH3LatentAV`; music nodes in `nodes_minimax_music.py`.

## Model catalog (researched live from HF, Oct 7 — registry must re-query, not hardcode)

**`Comfy-Org/MiniMax-H3`** (public, Comfy-repacked):
- DiT FL2VA ×6: bf16 66.3 G, int8_convrot 34.0 G, pruned_bf16 40.2 G, pruned_fp8 21.0 G,
  **pruned_int8 21.0 G (current)**, pruned_w6a8 16.0 G
- DiT Ref2VA ×6 (same size ladder) — reference mode (≤9 images/≤3 videos/≤3 audios, ≤12 total)
- Text encoders ×3: bf16 51.5 G, int8 27.1 G, **nvfp4_awq 15.7 G (current; works on non-Blackwell)**
- VAEs: video fp16 5.2 G / **int8 2.8 G (current)**; audio fp32 0.6 G (current)
- LoRAs ×3: **turbo 4step v1.0 768p** 1.96 G, turbo 8step v1.0 (template default) 1.96 G,
  ref2v turbo 4step 1.96 G
- ControlNets (`model_patches`) ×4: fun_controlnet_union 2.0 bf16 8.4 G / int8 4.5 G, union 1.0 bf16 4.2 G / int8 2.3 G
- Embeddings ×10: art_is_explosion, blooming_flowers, bullet_time, dark_magic, fire_breath,
  four_seasons, kiss_camera, spiral_ascent, storm_magic, truman_show

**Upstream / community (60+ repos found; registry prints live search `MiniMax-H3`):**
- `lightx2v/Minimax-h3-Turbo` (1.5 M dl) — turbo source; has **v1.0/v1.1/v1.2 768p** (Comfy-Org
  only carries v1.0), plus ref2v 4/8-step and fp8
- `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` — dense 1.49 G + VSA adapters (diffusers format)
- LoRAs: fal Realism-People, 360-Orbit, Character-Swap, wushu/spatial-physics/camera-motion,
  1980s-fantasy, Detail-VAE (`speach1sdef178`), latent upscaler (`LBH-123-AI`), fused turbo int8
  (`MATLOWAI`), pdmd 2/4-NFE (`Iwannapose`), FaceSwap REF2VA
- Quants: `unsloth/MiniMax-H3-GUF`, Abiray Pruned-GGUF, molbal GGUF, QuantFunc 4-bit
- Sources per Comfy README: `MiniMaxAI/MiniMax-H3` (gated original, 61.7 G-class BF16 partitions
  FL2VA+Ref2VA, shared Qwen3-VL encoder), lightx2v, `alibaba-pai/MiniMax-H3-Fun-Controlnet-Union`,
  `Kijai/MiniMax-H3-experimental`
- **MiniMax H3 Max** = fal.ai post-train, 480P/768P, faster (API product; Comfy template names exist,
  repo location TBD — research item)

**H3 itself (upstream facts):** 2 checkpoints FL2VA / Ref2VA; system = H3-Context-IR (prompt
enhancer) + H3-Base (768p AV) + H3-Regenerate-2K; 24 fps, 32 kHz stereo, 4–15 s, ratios 21:9…9:16;
prompt up to 7000 chars; tasks: t2va / fl2va / ref2va.

## Kaggle environment facts

- 2× T4 16 GB, **31.3 GB RAM** (→ all pinned weight caches OFF), `PHB` topology (host exchange
  ≈10 GB/s > p2p 9.3), 12 h session, **2 concurrent GPU sessions**, no interactive terminal,
  CLI logs empty while running (rely on notebook output).
- `/tmp/comfy.log` = Comfy stdout; wait cell tails it every 120 s as `[Ns] …`, plus `[h3ms]`
  (diagnostics) and `[gpu]` (nvidia-smi -l 10 → `/tmp/gpus.log`) lines.
- Dual-GPU proof protocol: `[GPUs] dit: 2 rank(s): cuda:0 (primary), cuda:1` +
  `[MultiStream] active: 2 ranks …` + both cards ~100 %/66 W; **fail fast** on `UNSPLIT` /
  `running unsplit on 1 GPU`; final assertion raises unless `active: 2 ranks` (v13 verified ✅).
- Measured performance: **861 s/step** (20-step dense, 0.98 MP, 124 frames) ≈ 4.8 h sampling vs
  pack reference 49.3 s/step on 2×5060 Ti — perf diagnosis pending step-time lines.
- Kaggle Dataset model cache: notebook must scan `/kaggle/input/**` → symlink-mirror into
  `models/<cat>/` → HF fallback → write `MANIFEST.json`; write-back to the dataset must **skip
  silently if the dataset is ≥200 GB full**.

---

## Phased plan (proper to-do)

### Phase A — v14: dynamic foundation + fast verified smoke (✅ VERIFIED ON KAGGLE)
1. ✅ **Registry cell**: live HF tree + community search → grouped catalog with
   `local/cached/remote` status — new files/repos auto-list every boot
2. ✅ **Settings cell**: `PRESETS` + `ACTIVE` (`fast_smoke`: turbo ON, 4-step lora, 0.4 MP =
   864×480, duration 5, short prompt, seed) — single source of truth for prompt + downloads
3. ✅ **Assets rework**: preset-selective download, local → `/kaggle/input` symlink mirror → HF
   (token), `MANIFEST.json`, per-file `OK <bytes>`
4. ✅ Converter applies `ACTIVE` (instance widgets + ResolutionSelector); wait cap → 3 h;
   dual-GPU verification retained
5. ✅ Local CPU-Comfy dry-run of the turbo prompt → **`/prompt` accepted** → pushed v14
6. ✅ **v14 run COMPLETE (25.6 min, zero errors)**: 5/5 files OK (41.03 GB) → turbo armed
   (`PrimitiveBoolean ['1021']`) → `prompt_id 67580883…`, `node_errors {}` → `active: 2 ranks
   cuda:0+cuda:1, 50 blocks, heads 28/28` → **209.6–210.9 s/step × 4** (4.1× vs 861 s/step dense)
   → MP4 `MiniMax_H3_00001_.mp4` → READY block with Base URL printed; exchange ≈6 % of step
   time ⇒ dense slowness is T4 compute, not split overhead

### Phase B — enterprise hybrid API (v15/v16) — ✅ VERIFIED ON KAGGLE
1. ✅ Sidecar cell (`api`): stdlib `ThreadingHTTPServer` on `:8190` serves `/h3api/*`
   **and** reverse-proxies everything else to Comfy `:8188` — raw bidirectional WebSocket pipe,
   chunked bodies, hop-by-hop header strip, CORS; one tunnel carries Comfy UI + native API + hub;
   `H3_API=0` disables (pub falls back to direct `:8188`)
2. ✅ Endpoints all live: `/h3api/` index, `/docs` (HTML), `/health`, `/catalog[?refresh=1]`,
   `/models`, `/settings` GET/POST, `/download` GET/POST, `/select` (download-on-select +
   **prune superseded `.safetensors`**, `prune:false` opts out), `/generate` (202 + prompt_id,
   `first_frame`/`last_frame` data-URLs), `/jobs`, `/outputs`, `/free`, `/gpus`,
   `/import?path=…` (raw-bytes streaming upload, disk guard, traversal rejected)
3. ✅ JSON envelope `{ok: true|false, ...}` everywhere + CORS; unknown route → 400/404 JSON
4. ✅ Docs: `GET /h3api/docs` HTML reference served in-process (repo API doc folded into GUIDE)
5. ✅ **sync cell**: `H3_CACHE_UPLOAD=auto|always|dry|never` — hardlink staging, manifest-drift
   probe, slug→title dataset discovery, `create`/`version` via `dataset-metadata.json`,
   **200 GB cap + disk guard auto-skip**, whole cell try/except ⇒ never fails the run
6. ✅ Shared `build_prompt(cfg)` extracted into `convert` cell (one builder, smoke + API both
   call it); first/last-frame `LoadImage` nodes 9001/9002 → subgraph input injection
7. ✅ Local dry-run **ALL TESTS PASSED**: every endpoint, WS `101`, chunked POST, import `201`,
   traversal `400`, prune verified, `first_frame` → 24-node prompt, sync `dry` staged 6 files
8. ✅ **v15 run captured (24.9 min, zero errors)**: hub listening `:8190`, tunnel verified
   (`https://costa-ltd-kelly-hosting.trycloudflare.com`), READY block with all `/h3api/*` URLs,
   smoke `prompt_id 59047396…`, 207 s/step × 4, `DUAL-GPU CONFIRMED`, MP4 out; **sync skipped**
   (`cannot determine dataset owner`) → **v16 built**: `kagglehub.whoami` owner + native
   `dataset_upload` + auth diagnostics + pub-cell live endpoint self-test (9 GETs + POST through
   the tunnel); dry-run ALL TESTS PASSED
9. ✅ **v16 run captured (32.9 min, zero errors)**: `in-kaggle-notebook: True` →
   `cache owner: adityahalde8777 (via kagglehub.whoami)` → `staged 5 model files` →
   `kagglehub upload OK -> adityahalde8777/minimax-h3-model-cache (530 s)` (42 GB created);
   self-test 7/9 `200` + expected `404` + `200 POST` — `/outputs`+`/gpus` URLError (tunnel
   transient; v17: retry + longer timeout + print reason). Smoke 194–196 s/step × 4, dual-GPU,
   MP4 out → **Phase B VERIFIED**

### Phase C — modalities (v16+, one low-quality smoke each, in order)
1. 🟡 **Image generation + use-case tasks (v17 tested → v18 healed+COMPLETED → v19 diagnostic rerun)** —
   **v17 run result (partial pass)**: image smoke produced `h3_image_00001_.png`, video smoke
   `MiniMax_H3_00001_.mp4` @ 204–206 s/step × 4 dual-GPU, `SMOKE PASSED`, pub self-test
   **11/11 `200`** (incl. the two v16 stragglers), tasks `upscale`/`video_frames` green —
   **bg_remove/extract failed**: `ImportError cannot import name '_slice' from
   'numpy._core.umath'` = **mixed numpy dir on the image after pip shuffled deps** (newer
   `strings.py` + 2.1.3 `umath.py`; `import numpy` fine, scipy/rembg dead) → pub raised →
   sync skipped. **v18**: install-cell probe + `force-reinstall --no-deps numpy==<ver>` heal +
   `TASKS_HEALTHY` gate (bg_remove/extract warn-only when unhealthy, upscale always required).
   **v18 run result (FIRST COMPLETE)**: subprocess probe `task deps health: OK` (no mixed numpy
   this boot) but **in-kernel verify FAILED** on `AttributeError: 'numpy.ufunc' object has no
   attribute '__module__'` → `TASKS_HEALTHY=False`. Root cause = **self-inflicted** v18 bug:
   `_kernel_verify` popped `numpy*` from `sys.modules` and re-imported in the same process —
   pop+reimport is itself broken (locally reproduced: `ImportError: cannot load module more
   than once per process` on 2.4.6; on Kaggle the re-import walks the newer-style
   `multiarray._override___module__` and dies setting `ufunc.__module__`). Result: warm-up,
   api `import rembg`, and every task re-imported the now-poisoned numpy → bg_remove/extract
   down for the whole boot. The **`TASKS_HEALTHY` gate absorbed it**: upscale (REQUIRED) green,
   video_frames green, bg_remove/extract `WARN tolerated` → `TASK SMOKE PASSED` → **sync RAN**:
   `inventory 62 files, 59.3 GB` → `manifest drift: 5 remote vs 6 local -> upload` →
   `kagglehub upload OK (716 s)` → **dataset version 2** → run **COMPLETED**. **v19 fix**:
   `_kernel_verify` non-destructive (one attempt, prints first error + traceback tail, no
   sys.modules surgery) + `_post` retry-once helper in pub (settings POST + task POSTs; v18's
   bg_remove died client-side with `Network is unreachable` tunnel blip).
   **Design (as shipped in v17, unchanged by v19)**: template `flux_schnell.json` (Comfy-Org/workflow_templates) =
   9 flat nodes, **no subgraph**, KSampler `4 steps, cfg 1, euler/simple`, latent 1024²,
   `CheckpointLoaderSimple` → **`Comfy-Org/flux1-schnell` / `flux1-schnell-fp8.safetensors`
   (17.24 GB, ONE file at repo root → downloaded into `models/checkpoints/`)**:
   - `build_prompt(cfg)` branches on `cfg["modality"]` → `_build_image` (separate flat converter
     reusing `load_info()` schema filter; Note/MarkdownNote skipped; `widgets_values_named`
     set for prompt (positive CLIPTextEncode via KSampler `positive` link origin), empty
     negative, ckpt/latent/sampler/seed/steps/cfg); no MultiStream/no turbo on image.
   - Preset `image_smoke` (1024², 4 steps, cfg 1, euler/simple, `h3_image` SaveImage prefix);
     `required_files` branches (`checkpoints` category); `PRUNE_DIRS` += `checkpoints`
     (selecting video prunes flux and vice versa).
   - Smoke schedule `H3_SMOKE="image,video"` (default; override to `"video"` to skip flux):
     image first (fast fail → PNG), wait cell polls a **`SMOKE_JOBS` list** with per-modality
     budgets (image 1800 s / video 10800 s) and **modality-aware assertions** (dual-GPU proof
     only when the video smoke ran; image smoke must produce a PNG).
   - Boot download ≈ 58 GB (41 GB video + 17.24 GB flux) + task assets (u2net 176 MB, ESRGAN
     67 MB) — `/tmp` has 1070 GiB on Kaggle; write-back drifts → second dataset upload expected.
   - **Use-case task engine** `GET /h3api/tasks` + `POST /h3api/task` (8 tasks):
     `bg_remove` (rembg → transparent PNG; default `u2net`, models: u2net/u2netp/
     isnet-general-use/u2net_human_seg/birefnet-*; `REMBG_HOME=/tmp/ComfyUI/models/rembg` so
     models land inside `models/` and get dataset-cached), `extract` (element extractor:
     rembg mask → `scipy.ndimage` 8-connected components → top-N element crops + bboxes +
     optional mask), `upscale` (RealESRGAN_x4plus.pth via Comfy `UpscaleModelLoader` +
     `ImageUpscaleWithModel`, synchronous history poll ≤300 s), `video_frames`, `video_gif`,
     `audio_extract`, `audio_trim`, `probe` (ffmpeg/ffprobe, `shutil.which` + `/tmp/bin` fallback,
     graceful 503 if missing). Inputs = data URLs or `<key>_path` (confined to `/tmp/ComfyUI`).
     Uniform envelope `{ok, outputs:[dataURL], saved:[names], view:[/view URLs], meta}`;
     ≤25 MB payloads fall back to saved/view URLs.
   - **Pub self-test**: GET probes get retry-once + 30 s timeout + `e.reason` (v16 blind
     `URLError` fixed), `/h3api/tasks` added, then a **task smoke through the tunnel**
     (bg_remove/extract/upscale on the newest PNG = required, video_frames on the smoke MP4 =
     warn-only) → prints `TASK SMOKE PASSED`.
   - Local CPU dry-run **59/59 ALL TESTS PASSED** (incl. real upscale 256→1024², alpha extent,
     bbox sanity, path-escape 400, sync dry). Upgrade path: Flux2 klein 4B distilled, Qwen-Image
     20B too big for T4
2. 🟡 **Music generation — v20 IMPLEMENTED & PUSHED** — **researched ✅** (see below) and built:
   template `audio_minimax_music_3.json` = subgraph "Text to Music (MiniMax Music 3)" (11 inner
   nodes) — flattener applies (**shared `_flatten(wf, info)` extracted from the video path;
   video prompts byte-identical after the refactor**). Caveat found and handled: unlike the H3
   video template, the music inner nodes store **positional `widgets_values` only** (older graph
   format, no `widgets_values_named`) — promoted by the **schema-verified `MUSIC_WIDGETS`**
   table (checked against every node's real `object_info` schema in a cloned ComfyUI):
   `UNETLoader [unet_name, weight_dtype]`, `MiniMaxMusic3TextEncode [caption, lyrics, seed, "",
   max_duration, cfg_scale, top_k]` (the `""` skips the frontend-only `control_after_generate`
   slot), `CLIPLoader [clip_name, type, device]`, `VAELoader [vae_name]`,
   `EmptyMiniMaxMusic3LatentAudio [seconds, batch_size]`, `KSampler [seed, "", steps, cfg,
   sampler_name, scheduler, denoise]`, `VAEDecodeAudioTiled [tile_size, overlap]`,
   `SeedNode [seed]`, `ComfySwitchNode [switch]`. Node set is **ComfyUI core**
   (`comfy_extras/nodes_minimax_music.py`, `nodes_audio.py`, `nodes_logic.py`) — **no custom
   pack, no MultiStream, single T4** (your one source of 2-GPU proof stays the H3 video lane).
   Wiring verified node-by-node: instance `switch` → ComfySwitchNode selects **VAEDecodeAudioTiled**
   (tiled) vs VAEDecodeAudio (lazy inputs — only the chosen one executes); seed → SeedNode →
   TextEncode.seed + KSampler.seed; `max_duration` → TextEncode + EmptyLatentAudio.seconds;
   subgraph output link 62 → ComfySwitchNode out slot 0 → **SaveAudioAdvanced.audio**
   (`format` is an **`IO.DynamicCombo`** — the API prompt must carry the option key as a PLAIN
   STRING `"flac"`; the executor matches the option, injects its sub-inputs (mp3/opus →
   `quality`) and re-nests `{"format": ...}` at call time. v20 sent the dict
   `{"format":"flac"}` → silent option-key mismatch → input dropped →
   `missing positional 'format'`; **fixed in v21**). Preset
   `music_smoke` = max_duration 5, seed 4242, tiled_decode True, format `flac`, prefix `h3_music`;
   models from **`Comfy-Org/MiniMax-Music-3`**: dit fp16 **4.91 GB** + TE pruned int8 convrot
   **9.20 GB** + dav VAE **0.22 GB** = **14.33 GB** (also listed: dit fp32 9.8 / dit int8 2.5 /
   TE bf16 18.5 / TE pruned bf16 16.7); all land in the same `diffusion_models/text_encoders/vae`
   dirs the registry + prune already cover; `snapshot_download(local_dir=models)` (repo paths are
   already category-prefixed); outputs via `SaveAudioAdvanced` → `/tmp/ComfyUI/output/h3_music_*.flac`.
   `H3_SMOKE` default = **`image,video,music`**; wait cell adds `BUDGET["music"]=3600` + scans
   `auds` for the audio file ("music smoke finished but produced no audio output"). Registry cell
   adds the music catalog + `music_repo`; api docs/catalog carry music rows; `required_files`
   gets the music branch. Local gate before push: **video byte-equivalence PASS** (old inline
   flatten == new `_flatten`, 27 nodes, identical meta), **music build OK** (14 nodes,
   structural checks PASS, SaveAudioAdvanced wired, seeds routed) — done via an equivalence
   harness with a mocked object_info.
3. ⬜ **Transcription** — **researched ✅** (no official local template; only cloud `api_*`
   STT). Front-runners: **`Setmaster/comfyui-stt7`** (MIT, purpose-built suite:
   LoadModel/TranscribeAudio/LoadAudioFile, faster-whisper, built-in VRAM mgmt) and
   **`yuvraj108c/ComfyUI-Whisper`** (275★, auto-downloads models, license NOASSERTION ⚠);
   MIT alternatives: `endman100/ComfyUI-WhisperLargeV3-Repack`. Smoke input = a short
   speech WAV (downloaded sample) → native `LoadAudio` → whisper node → text; decision: pick
   one pack, pip-install its requirements in the install cell
4. ⬜ **Soundtrack**: H3 native joint audio (already produces 32 kHz stereo) — document + API field
5. ⬜ Each modality gets: registry entries, settings schema, docs section, one 480p/low smoke

### Phase D — community model integration
1. ⬜ Ref2VA video lane (`api_minimax_h3_r2v.json` + ref DiT + ref2v turbo LoRA) → smoke
2. ⬜ First/last-frame I2V via API (image upload → `first_frame`/`last_frame`)
3. ⬜ Community imports: lightx2v v1.1/v1.2 turbos, style LoRAs, ControlNet, detail VAE, upscaler
4. ⬜ `/h3api/import {repo, files}` for arbitrary models; catalog auto-grows

### Phase E — housekeeping
1. ✅ **`build_full.py` moved into this repo** (v19) — reconstructed verbatim from the v18
   notebook after the host restart wiped `/tmp/opencode`; rebuild verified byte-identical against
   `git show HEAD:minimax_h3_full_comfy.ipynb`; cells embedded as raw `r'''…'''` blocks, variable
   names = `title_md, secrets, preflight, install, registry, settings, assets, convert, start,
   api, smoke, wait, pub, sync`; writer emits `source:[src]` single-element lists
2. ✅ v13/v14 outcome: step-time diagnostics (`[MultiStream] step:`) + READY block captured in
   both runs (v13 dense 861 s/step, v14 turbo 210 s/step)
3. ✅ Perf explained (v14): exchange ≈6 % of a step ⇒ **T4 compute-bound**; levers landed = turbo
   + 0.4 MP (4.1× per step); untested levers = VAE-split + TE cache
4. 🟡 **Dataset cache bootstrap partially done**: v16 created it (42 GB), **v18 created version 2**
   (nvfp4 15.7 G + turbo lora + MANIFEST + placeholders), and **v19 CONFIRMED the write-back half
   idles**: `cache dataset up to date (manifest match)` (no drift → no upload). The read half
   still needs the **one-time manual step**: attach `minimax-h3-model-cache` via Add Data so
   `/kaggle/input` populates (v19 still showed `0 candidate files` → boots still download from HF)
5. ⬜ Rebuild `dryrun.py` into the repo (lost with the host restart) so v20+ keeps the local
   59/59 gate
6. ⬜ Rotate the GitHub PAT pasted in chat earlier; keep HF token out of the public repo

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| Oct 7 | Replaced WanGP/FastAPI/Cloudflare-API stack with ComfyUI + H3 MultiStream | Research showed 2×16 GB PCIe dual-GPU H3 works (~1.76× dense / ~2.25× sparse); single-lane = simpler ops |
| Oct 7 | Model files from `Comfy-Org/MiniMax-H3`, not DeepBeepMeep | Matches the official Comfy template's widget names exactly |
| Oct 7 | `exchange="host"`, `exchange_chunks=8` | Measured on Kaggle: host ≈ 10 GB/s vs p2p ≈ 9.3 GB/s, topology PHB (no NVLink) |
| Oct 7 | Never `pip install -U huggingface_hub` in the notebook | v3: bump to 2.1.x broke ComfyUI's pinned gradio/tokenizers/diffusers → server never listened |
| Oct 7 | Submit workflows in **API prompt format**, never the saved UI JSON | v5: `{"nodes":[...],"links":[...]}` → HTTP 500; `/prompt` wants `{id:{class_type,inputs}}` |
| Oct 7 | Prompt keys **and** link ids as strings (`["4", 1]`) | `validate_inputs` does `prompt[val[0]]` with JSON keys (always str) |
| Oct 7 | Skip UI-only node types, verified against `GET /object_info` | v8: `MarkdownNote` is frontend-only → `missing_node_type` 400 |
| Oct 7 | **Flatten subgraphs** instead of skipping them | v10: the H3 pipeline lives inside `definitions.subgraphs[0]`; skipping left `SaveVideo` linking to a missing node → `KeyError` 400 |
| Oct 7 | Drop frontend-only widget keys (`fixed`, `control_after_generate`, …) | Schema-aware filter against `object_info` keeps only real inputs |
| Oct 7 | Download `vae/minimax_h3_video_vae_int8_convrot.safetensors` (not fp16) | Template widget names the int8 file → COMBO `value not in list` otherwise |
| Oct 7 | `megapixels=0.98` on `ResolutionSelector` | Template default 0.4 MP = 864×480; 0.98 = 1344×768 official 768p (user wants ~1K) — *superseded for smoke by fast preset (0.4), kept as `quality` preset* |
| Oct 7 | `weight_cache=False` (and all pinned caches off) | Only **31.3 GB** system RAM on Kaggle; DiT cache alone = 18 GiB pinned |
| Oct 7 | POST `extra_data: {"preview_method": "latent2rgb"}` | Node-pack README: API clients must send it, the web UI sends it implicitly |
| Oct 7 | Insert `H3MultiStream` after the model switch → **BasicGuider + BasicScheduler** | README recommended wiring; bypasses only the turbo-lora branch selector |
| Oct 7 | Drop the separate "patch cell"; conversion + insertion live in the submit cell | Patch cell looked for `BasicGuider` in the top-level graph and never found it (inside the subgraph) |
| Oct 7 | `H3_SHUTDOWN=1` guard on cleanup cell | cleanup used to kill the server right after printing READY |
| Oct 7 | Only ONE kernel on the account | 2-session GPU cap; old lanes deleted |
| Oct 7 | Download via `huggingface_hub.snapshot_download(...)` + post-download size verification | v11: `python -m huggingface_hub.cli.download` has **no `__main__` guard** → exits 0, downloads NOTHING |
| Oct 7 | Always include **linked** inputs even if absent from the static `object_info` schema | v11: autogrow slots (`values.a`) are created only during validation finalization |
| Oct 7 | **Prove dual-GPU execution at runtime, don't infer it from config**: `nvidia-smi -l 10` → `[gpu]` lines + assertion on `[MultiStream] active: 2 ranks`, fail on `UNSPLIT` | config shows intent, not execution — verified live in v13 ✅ |
| Oct 7 | Ship the verification as **v13 after v12's outcome**, not mid-run | pushing triggers a fresh run → re-download + burns the 2-session cap while v12 is live |
| Oct 7 | **Keep quality config** for v13; speed work after dual-GPU proven | user: "keep quality, but this time dual GPU verify" |
| Oct 7 | Wait cap sized from **measured** throughput + fail fast on `UNSPLIT` | v12's 60-min cap would have false-failed a healthy 4.8 h job |
| Oct 7 | **Scope = all-in-one generation hub** (video/image/music/sound/transcription; NO text) behind one Base URL | user directive; H3 video is lane #1 of many |
| Oct 7 | **Smoke = one low-quality run per modality/category**, never every model | user: testing all models would be "worst"; goal is verifying the system |
| Oct 7 | First smoke: **4-step turbo LoRA @ 0.4 MP (864×480)**, quality graph superseded | user: fast + verified dual-GPU run |
| Oct 7 | **Dynamic registry**: live HF query each boot → auto-list new models; download-on-select; unload/purge previous (Comfy `/free`) | user-described pattern (select → download → previous gone → always clean/fast) |
| Oct 7 | **Dataset cache with 200 GB cap guard**: write-back auto-skips when full, never fails | user directive; Kaggle dataset limit |
| Oct 7 | HF token: Kaggle Secret first, runtime-assembled fallback; never in public git | user provided token for fast downloads; repo is public |
| Oct 7 | **No `negative_prompt` in H3 API schema** | H3 is CFG-distilled (CFG 1); source has prompt-only conditioning |
| Oct 7 | Enterprise API = sidecar `/h3api/*` + proxy (WS passthrough) on the same tunnel as Comfy | one Base URL for Comfy UI, native API, and the management API + docs |
| Oct 7 | Hub + proxy in **one stdlib `ThreadingHTTPServer` on `:8190`** (in-process, no extra process); tunnel that port; `H3_API=0` kill-switch falls back to `:8188` | one port/one tunnel/no supervision; stdlib = zero install risk on the kernel image |
| Oct 7 | Raw socket **bidirectional WebSocket pipe** + hop-by-hop header strip + chunked-body reader in the proxy | Comfy UI's `/ws` must survive proxying; `http.client` can't do WS upgrades |
| Oct 7 | `/h3api/select` **prunes superseded `.safetensors` by default** (`prune:false` opts out), only in the 4 model dirs | user pattern: switching models must free disk + GPU, never bloat |
| Oct 7 | Extract `build_prompt(cfg)` into a `convert` cell — smoke **and** API both call it | one builder, two callers: the dry-run prompt and the live `/generate` can't drift apart |
| Oct 7 | `/generate` accepts `first_frame`/`last_frame` as **data-URLs** → written to disk → `LoadImage` 9001/9002 injected at subgraph inputs 0/1 | keeps the API self-contained (no file-upload round trip needed for conditioning) |
| Oct 7 | Write-back cell wrapped **entirely** in try/except with `auto\|always\|dry\|never` policy; hardlink staging; manifest-drift probe; 200 GB + disk guards auto-skip | user directive: cache-full must auto-skip and **never fail the run** |
| Oct 7 | Notebook-cell escapes **doubled** in the build script (`b"\\r\\n"` → `b"\\\\r\\\\n"`) + `ast.parse` every *generated* cell | v15 pre-push: the outer triple-quoted string ate the backslashes → generated cell had an unterminated bytes literal (build script itself parsed fine) |
| Oct 7 | Kaggle dataset CLI: `version -p FOLDER` (reads `dataset-metadata.json`; **no `-d`**), `list -s X -v`, existence probe = `datasets metadata <ref>` | verified against the installed CLI while wiring the sync cell |
| Oct 7 | Write-back auth = **kagglehub first** (`whoami` for owner, `dataset_upload` to push), classic CLI second, **embedded kernel owner** (public, build-time from kernel-metadata `id`) last | v15 run proved modern notebooks carry a **token file** (`KAGGLE_API_V1_TOKEN`), not `KAGGLE_USERNAME`/`kaggle.json` — and the in-kernel CLI got `403`; kagglehub has native notebook auth |
| Oct 7 | Sync cell prints **auth diagnostics** every run (env/file booleans + `in-kaggle-notebook`, never secrets) | v15's skip was silent-adjacent: next run's log must show which cred sources exist without debugging blind |
| Oct 7 | Pub cell **self-tests the public surface** (health/catalog/settings/jobs/outputs/gpus/docs, proxied `/object_info`, 404 envelope, no-op POST settings) through the tunnel before READY | tunnel-up ≠ API working; logs must *prove* every endpoint end to end (v15 had only `/history` + `/health` checks) |
| Oct 7 | Self-test GETs get **retry-once + 30 s timeout + print `e.reason`** (v17) | v16: `/outputs`+`/gpus` returned bare `URLError` — reason was swallowed, so a tunnel stall and a slow handler were indistinguishable |
| Oct 7 | **Local caches redirected to `/tmp`** (`XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `HF_HOME`, npm cache, `TMPDIR` in `.bashrc`/`.profile`) | home partition is only 4.8 GB and hit 96%; `/tmp` lives on the 119 GB root disk |
| Oct 7 | **Colab CLI session (`hub`) = remote compute**; home directory holds project files only (user directive) | heavy jobs (installs, dry-runs, tests) must not consume home disk; Colab gives 107 GB + torch for free |
| Oct 7 | **Image lane = separate `_build_image(cfg)`**, not the H3 subgraph flattener (v17) | `flux_schnell.json` is already flat (9 nodes); keeping the verified video path untouched is worth ~30 lines of dedicated converter |
| Oct 7 | **`H3_SMOKE` schedule (default `image,video`, image first)** + `SMOKE_JOBS` list + per-modality budgets (1800 s / 10800 s) + **modality-aware assertions** (dual-GPU proof only if video ran; image must yield a PNG) | user rule: one low-quality smoke per modality; image first = fastest fail signal and its PNG feeds the task self-test; flux runs single-T4 so the `active: 2 ranks` assertion must not fire on image-only boots |
| Oct 7 | **Task engine = in-process for CPU work (rembg + scipy + ffmpeg), GPU only via Comfy** (`UpscaleModelLoader` → synchronous `/history` poll ≤300 s) | keeps tasks fast and dependency-light; only upscale needs the GPU; a Comfy-executed upscale reuses the existing validated queue path |
| Oct 7 | Pin **`rembg[cpu]==2.0.85`**, default model **`u2net`** (explicitly not rembg's own default `bria-rmbg`) | 2.0.85 verified locally (new `new_session`/`REMBG_HOME` API); bria-rmbg = 1.02 GB download + SIGKILL on the 8 GB local box + license ambiguity; u2net = 176 MB, well-understood license |
| Oct 7 | `REMBG_HOME=/tmp/ComfyUI/models/rembg` | segmentation weights must live **inside `models/`** so the inventory → hardlink staging → dataset cache picks them up automatically |
| Oct 7 | Flux root file downloaded with `snapshot_download(..., local_dir=models/checkpoints)`; `IMG_FILES` set shared by boot, `/h3api/download`, and prune | repo stores `flux1-schnell-fp8.safetensors` at the ROOT, but Comfy's combo lists category dirs — relocate at download time; prune `checkpoints` like the other model dirs |
| Oct 7 | Task response envelope: `{ok, task, outputs:[data URLs], saved:[names], view:[/view URLs], meta}` with a **25 MB payload cap** falling back to saved/view | data URLs are the easiest client UX; unbounded base64 (4096² upscale = ~55 MB) would make responses unpredictable |
| Oct 7 | Pub self-test **requires** bg_remove/extract/upscale to pass, video tasks warn-only | image tasks are the core v17 feature (fail the run loudly); ffmpeg availability varies by image, so video tasks must not fail a healthy run |
| Oct 7 | Local dry-run recreates 12-byte weight placeholders **at boot and before the image generate test** | the select-prune test legitimately deletes `checkpoints/flux…` mid-run (Comfy re-scans the combo list), and only 11 GB disk is free locally — never let assets see a missing flux file |
| Oct 7 | **v18: install-cell numpy health probe + auto-heal** (`from numpy._core.strings import *; from scipy import ndimage; import rembg` in a subprocess, then `pip force-reinstall --no-deps numpy==<ver>` on failure, plus numpy diagnostics: `__path__`, dist-info list, per-file `_slice` markers) and an **in-kernel verify** that drops stale `numpy*` modules and retries | v17 died after everything else passed because a **mixed numpy dir** silently kills scipy/rembg while `import numpy` still works; the failure only surfaced in the *task smoke*, 31 minutes in — probe the exact imports you depend on, at install time, and heal from inside the notebook (a Kaggle boot cannot be debugged interactively) |
| Oct 7 | **`TASKS_HEALTHY` gate on the pub task smoke**: bg_remove/extract REQUIRED only when the probe+heal succeeded (warn-only otherwise), upscale always required, video tasks warn-only; `/h3api/tasks` exposes `healthy` + `rembg_error` | a secondary feature (bg_remove) must not block READY and the write-back cell for the whole stack when the environment is the problem — but a *healthy* environment must prove the feature works; the run still completes and the log still shows the exact import error |
| Oct 7 | **Always print the pip tail, even on rc=0** (was: print output only on failure) | v17's discarded pip output hid whether the rembg resolve moved numpy — the single most important diagnostic line was thrown away by a "keep logs clean" default |
| Oct 7 | Cell-order fact: **sync only runs if pub succeeds** — a pub raise silently skips write-back | gates before READY must be calibrated to *environment-dependent* features; core-generation failures stay fatal, environment-conditional ones degrade with a loud warning |
| Oct 8 | **v19: `_kernel_verify` is non-destructive — ONE attempt, print the FIRST error + traceback tail, return False, NEVER touch `sys.modules`** | v18's pop-and-reimport verify was self-inflicted damage: removing `numpy*` from a long-lived kernel and re-importing is broken by itself (locally: `cannot load module more than once per process`; on Kaggle: `ufunc.__module__` AttributeError) — a small install hiccup escalated into total in-kernel numpy loss for the whole boot. The useful behavior is diagnosis (log what actually failed) + graceful `TASKS_HEALTHY=False` |
| Oct 8 | **v19: all pub POSTs (settings + task smoke) share one `_post` helper with retry-once + `e.reason` logging** | v18: `bg_remove` task request died client-side with `Network is unreachable` (tunnel blip) and `POST /h3api/settings` failed on the single attempt — GET probes already retry, the POSTs must too, so a healthy state can't false-fail |
| Oct 8 | **Kaggle Dataset v18 outcome keeps versioning as-is**: manifest drift probe is authoritative — a populated dataset makes later boots idle on `manifest match` and upload only on real drift | v18 uploaded version 2 (716 s) from a `5 remote vs 6 local` drift; no manual seesaw of inventory counts needed |
| Oct 8 | **v19 root cause, now behind the v20 heal**: the v17/v19 mixed numpy is an **in-place numpy swap during the install's pip subprocesses** — `pip install rembg[cpu]` re-resolves numpy and replaces the wheel files while THIS kernel has already pre-loaded numpy 2.1.3 (torch in preflight). Disk becomes self-consistent (fresh subprocess probes pass on the new version) but the kernel's loaded `umath` (old) + new on-disk `strings.py` mix → `_slice` ImportError. Locally reproduced 1:1 (2.1.3 preload → in-place upgrade to 2.5.3 → exact same in-kernel error); **cure proven**: `pip force-reinstall --no-deps numpy==<kernel-loaded version>` then both probes OK | the subprocess-only probe design had a blind spot: it validates disk files, not this process's loaded modules. The heal must resync the wheel to the version the KERNEL loaded (`numpy.__version__` in-process), not to whatever subprocess `pip` put there |
| Oct 8 | **v20: heal fires when EITHER probe fails** (subprocess OR in-kernel), re-pins numpy to the in-kernel loaded version, re-verifies both, then `TASKS_HEALTHY` | v19 proved the divergence: subprocess OK + in-kernel FAIL. Trusting only the subprocess means never healing the actual broken thing |
| Oct 8 | **v20 music lane = ComfyUI core nodes only** (`nodes_minimax_music.py` + `nodes_audio.py` + `nodes_logic.py`), single T4, no MultiStream | MiniMax Music 3 has no custom pack; it mus not disturb the only source of 2-GPU proof (H3 video). Core nodes are schema-verified from a ComfyUI clone — no guesswork |
| Oct 8 | **v20 music template uses positional→named widget promotion (`MUSIC_WIDGETS`)**, not `widgets_values_named` (the music subgraph is the older format) | 11 node schemas verified from source; the `""` slots skip `control_after_generate`, linked inputs always override — the flattener's shared logic then applies unchanged |
| Oct 8 | **v20 `_flatten(wf, info)` extracted from the video flattener**; music calls the same helper | video prompts must stay byte-identical after the refactor — **enforced by an equivalence test** (old inline flatten vs `_flatten` → identical 27-node prompt + identical meta) before push |
| Oct 8 | **v20 `H3_SMOKE` default becomes `image,video,music`**; music budget 3600 s, wait cell asserts a real audio file (`auds` scan), format `flac` | user rule: one low-quality smoke per modality; music smoke = 30-step single-T4 mini run |
| Oct 8 | **v21: `SaveAudioAdvanced.format` (and all `IO.DynamicCombo` inputs) is a PLAIN STRING in the API prompt** (`"flac"`), not the UI-widget dict `{"format":"flac"}`; mp3/opus set the dotted sub-input `format.quality`; `_flatten` now emits dotted keys whose parent input is present | v20 died on `execute() missing 'format'` — proved on a local CPU ComfyUI that the dict form silently fails the DynamicCombo option-key match and drops the input from the executor schema, while the string form validates clean and is re-nested to `{"format": ...}` by `build_nested_inputs`. Verify against the actual node schema, not the saved-workflow widget shape |
| Oct 8 | **v21: reproduce executor behaviour before pushing** (local CPU ComfyUI install: `get_finalized_class_inputs` + `build_nested_inputs` + real `/prompt`) | the v20 failure was invisible to `node_errors` validation (validation iterates the expanded schema only — a dropped dynamic input passes silently and dies at execute). The only reliable gate is the executor path itself |

---

## Task log

| Date | Task | Result |
|---|---|---|
| Oct 7 07:34 | Initial Kaggle workspace, project skeleton | Done |
| Oct 7 07:48 | v1 push (WanGP lane) | Failed: `pinned FL2VA not found` |
| Oct 7 | v2–v5 model-def / downloader patches | Progressed through: GenerationError masking → patched HTTP downloader → HF xet blocked |
| Oct 7 | v6 pre-download assets + `HF_HUB_DISABLE_XET=1` | ✅ 12 assets fetched; smoke entered |
| Oct 7 | MultiGPU probe v1–v3 | 2× T4 cc7.5 14.6 G, PHB; host ≈ 10.0–10.3 GB/s vs p2p ≈ 9.3–9.4 GB/s ✅ |
| Oct 7 | WanGP v1 end-to-end | ✅ READY printed, public URL verified, ≈ 80 min; cleanup killed server → guard added (v7) |
| Oct 7 | Added ERROR_PLAYBOOK.md, GUIDE.md, BUILD_YOUR_OWN.md, CONCLUSION.md | ✅ |
| Oct 7 | Switched repo to single Comfy 2×T4 lane; deleted all other kernels | ✅ only `minimax-h3-comfy-2xt4-generator` remains |
| Oct 7 | Comfy v1–v2 launcher | v2 finished 368 s but only *submitted* the prompt (no wait) |
| Oct 7 | v3 full pipeline | ❌ `RuntimeError: ComfyUI did not listen in time` — caused by `pip install -U huggingface_hub` |
| Oct 7 | v5 | ❌ removed hf upgrade (server starts) but `/prompt` → **HTTP 500** (saved UI JSON posted) |
| Oct 7 | v6 | ❌ API-format converter written → **HTTP 400**, body not printed (debug bug) |
| Oct 7 | v7 | ❌ `NameError: name 'e'` inside the error handler masked the 400 body |
| Oct 7 | v8 | ❌ 400 body finally visible: `missing_node_type: MarkdownNote` |
| Oct 7 | v9 | ❌ `json.load(bytes)` typo (`AttributeError: 'bytes' object has no attribute 'read'`) |
| Oct 7 | v10 | ❌ filter works → 400 `prompt_outputs_failed_validation`, `SaveVideo` `KeyError` on node 140 |
| Oct 7 | Inspected template + ComfyUI source + node-pack README | Found: whole pipeline is a **subgraph**; VAE filename mismatch; 0.4 MP default; 31.3 GB RAM; `preview_method` requirement; string link ids |
| Oct 7 | v11: subgraph flattener + schema filter + H3 insert + asset/resolution/RAM fixes | ❌ 6 validation errors: 5× `value_not_in_list` (model folders EMPTY) + `values.a` required missing (autogrow) |
| Oct 7 | Root-caused v11 locally | fixed: in-process `snapshot_download` + size check; links bypass schema filter |
| Oct 7 | v12 pushed | QUEUED — local dry-run green |
| Oct 7 | v12: assets OK (5/5, 41 GB), Comfy ready 45 s, **`/prompt` accepted** (`prompt_id b5f94a63…`, `node_errors: {}`) | ✅ first run to pass validation in this lane |
| Oct 7 | Investigated external GPU-check paths | impossible (no terminal, 127.0.0.1 bind, CLI logs lag) → pack logs + nvidia-smi monitor designed |
| Oct 7 | v13 built locally: monitor + `[gpu]` lines + hard dual-GPU assertion | ✅ `ast`-clean, waited for v12 outcome |
| Oct 7 | v12 live log analysed: **861 s/step**, ETA ≈4.8 h vs reference 42–49 s/step; 60-min cap would trip | user: terminate, keep quality, verify dual GPU |
| Oct 7 | v13 pushed (quality graph + verification suite + 6.5 h cap) | PUSHED |
| Oct 7 | **v13 dual-GPU verified in production**: `2 rank(s): cuda:0+cuda:1`, `active: 2 ranks, 50 blocks, 28/28 heads`, both GPUs `100 % / 65.9 W / 10.4+8.6 GiB`, no `UNSPLIT` | ✅ core requirement met; all six MDs updated (commit `30f7549`) |
| Oct 7 | **Deep research pass** (user order): Comfy-Org full tree (12 DiT / 3 TE / 3 VAE / 3 LoRA / 4 ControlNet / 10 embeddings), lightx2v turbo versions, 60+ community repos, Comfy native H3 nodes, official API-format templates (t2v/r2v/max/music), upstream H3 system (FL2VA/Ref2VA/Context-IR/2K) | catalog captured above |
| Oct 7 | Traced turbo wiring + resolution/duration/negative-prompt facts in template & Comfy source | one boolean flips LoRA+steps; 0.4→864×480; 17k+5 snapping; no negative prompt |
| Oct 7 | Started local CPU torch + ComfyUI install for offline `/prompt` dry-runs | in progress (`/tmp/opencode/pip.log`) |
| Oct 7 | AGENT.md rewritten as full master-context file (this) | Done |
| Oct 7 | v14 built (registry + settings + preset-selective assets + symlink cache) and pushed | PUSHED |
| Oct 7 | **v14 VERIFIED on Kaggle — Phase A complete**: run 25.6 min / zero stderr errors; 5/5 `OK` (41.03 GB, HF token, minutes); registry auto-listed 39 core + 100 community; `fast_smoke` applied; turbo armed `['1021']`; `prompt_id 67580883… node_errors {}`; `active: 2 ranks cuda:0+cuda:1, 50 blocks, 28/28`; **209.6–210.9 s/step × 4**; both T4s 100 % / 64.9+62.1 W; MP4 `MiniMax_H3_00001_.mp4`; READY Base URL `https://poem-smile-fell-ddr.trycloudflare.com` | ✅ all six MDs updated (commit `a9defb6`) |
| Oct 7 | v15 built (14 cells): `convert` (`build_prompt` + frame injection), `api` (hub `:8190` + proxy + WS), `sync` (dataset write-back); local CPU dry-run **ALL TESTS PASSED** (every endpoint, WS `101`, chunked, import `201`, traversal `400`, prune, sync `dry`) | ✅ |
| Oct 7 | v15 pushed to Kaggle (kernel version 15) | RUNNING — watching |
| Oct 7 | Six-MD docs pass for v14 verification + v15 hub API (README/GUIDE/CONCLUSION/ERROR_PLAYBOOK/BUILD_YOUR_OWN/AGENT) | commit `1e3a64b` |
| Oct 7 | **Phase C pre-research**: image = `flux_schnell.json` template + `Comfy-Org/flux1-schnell` fp8 single file (17.24 GB, 4 steps cfg 1, no subgraph); music = `audio_minimax_music_3.json` subgraph + `Comfy-Org/MiniMax-Music-3` (14.33 GB, 30 steps cfg 1.7); transcription = no official local template → shortlist `Setmaster/comfyui-stt7` (MIT) vs `yuvraj108c/ComfyUI-Whisper` (275★, NOASSERTION license) | findings recorded in Phase C section |
| Oct 7 | **v15 VERIFIED on Kaggle — Phase B verified**: 24.9 min, zero stderr; `hub API + proxy listening on :8190`; tunnel `Verified public endpoint: https://costa-ltd-kelly-hosting.trycloudflare.com` + full READY with `/h3api/*` URLs; `prompt_id 59047396… node_errors {}`; **207.2–208.2 s/step × 4**; `DUAL-GPU CONFIRMED`; MP4 out; **sync bug**: `SKIP upload: cannot determine dataset owner` (notebook auth = token file) | ✅ logs captured (`/tmp/opencode/v15_stdout.txt`) |
| Oct 7 | v16 built: sync owner via `kagglehub.whoami` + env/kaggle.json/embedded fallbacks + auth diagnostics; upload via `kagglehub.dataset_upload` (CLI fallback); pub live endpoint self-test (9 GET + POST through tunnel); dry-run **ALL TESTS PASSED** (incl. `cache owner: adityahalde8777 (via kagglehub.whoami)`) | pushed |
| Oct 7 | **v16 VERIFIED on Kaggle — Phase B complete**: 32.9 min / zero errors; `in-kaggle-notebook: True`; `cache owner: adityahalde8777 (via kagglehub.whoami)`; `staged 5 model files`; **`kagglehub upload OK -> adityahalde8777/minimax-h3-model-cache (530 s)`** (42 GB first-time create); self-test 7/9 `200` + `404 /h3api/nope` + `200 POST /h3api/settings`; smoke `prompt_id 1964b47f… node_errors {}`, **194.4–195.9 s/step × 4**, `DUAL-GPU CONFIRMED`, MP4 out; only `/h3api/outputs`+`/gpus` URLError (tunnel transient → v17 retry fix) | ✅ logs `/tmp/opencode/v16_stdout.txt`; all six MDs updated |
| Oct 7 | **Disk hygiene** (user: home dir 96% full): purged pip cache (919 MB) + npm cache (811 MB) → home 96%→59%; redirected `XDG_CACHE_HOME`/`PIP_CACHE_DIR`/`HF_HOME`/npm cache/`TMPDIR` to `/tmp` via `.bashrc`/`.profile` | ✅ |
| Oct 7 | **Colab = remote machine** (user directive: home only for files): Colab CLI session `hub` created + probed (`colab exec` runs code: Python 3.13, 107 GB disk, torch CPU); heavy compute (dry-runs, testing) goes to Colab, home keeps repo files only | ✅ |
| Oct 7 | **v17 local test assets**: `rembg[cpu]` + onnxruntime (u2netp/u2net warmed under `REMBG_HOME`), `RealESRGAN_x4plus.pth` (67 MB → `models/upscale_models/`), static ffmpeg+ffprobe 7.0.2 → `/tmp/bin`, 12-byte flux placeholder → `models/checkpoints/` | ✅ (rembg default bria-rmbg 1.02 GB SIGKILL'd the 8 GB box → switched to u2net) |
| Oct 7 | **v17 built** (image branch + task engine + smoke schedule + pub retry/task smoke) → `ast`-validated 14 cells → local dry-run **59/59 ALL TESTS PASSED** (incl. real `upscale 4× 256→1024²`, bg_remove alpha extent, extract bbox, 8 tasks, path-escape 400, pub `TASK SMOKE PASSED`) | ✅ two dry-run bugs fixed en route: `T_upscale` forgot the `{"prompt": …}` wrapper; image generate test ran after the prune test deleted the flux placeholder |
| Oct 7 | **v17 pushed** (kernel version 17) — image lane + use-case tasks + self-test retry | RUN **PARTIAL FAIL at 31 min**: image smoke PNG ✅, video MP4 @ 204–206 s/step × 4 ✅ dual-GPU ✅, `SMOKE PASSED` ✅, pub self-test **11/11 `200`** ✅ (outputs+gpus retry fix verified), tasks upscale/video_frames ✅ — but bg_remove/extract hit the **mixed-numpy `_slice` ImportError** → pub raised → sync skipped; root-caused from the downloaded log |
| Oct 7 | **v18 built** — numpy probe + auto-heal + in-kernel verify in the install cell, `TASKS_HEALTHY` gate in pub, `rembg_error`/`healthy` in `/h3api/tasks`, pip tail always printed → rebuilt, 13 cells `ast`-validated, dry-run **59/59** | ✅ |
| Oct 7 | **v18 pushed** (kernel version 18) — probe + heal + TASKS_HEALTHY gate | ✅ PUSHED |
| Oct 8 | **v18 run COMPLETED — FIRST GREEN END-TO-END**: `task deps health: OK` (no mixed numpy this boot); **in-kernel verify FAILED** (`AttributeError: 'numpy.ufunc' object has no attribute '__module__'` → `TASKS_HEALTHY=False`) — root-caused as **self-inflicted pop+reimport bug** + reproduced locally; `DUAL-GPU CONFIRMED`, `SMOKE PASSED: ['image','video']`; task smoke upscale REQUIRED ✅ + video_frames ✅, bg_remove/extract `WARN tolerated` → **`TASK SMOKE PASSED`**; **sync RAN**: `inventory 62 files, 59.3 GB` → drift 5 vs 6 → **`kagglehub upload OK (716 s)` → dataset version 2**; run **COMPLETED** (kernel done, no ERROR); foundation log `/tmp/opencode/v18out/` | ✅ all six MDs pass pending (commit this round) |
| Oct 8 | **v19 built**: `_kernel_verify` non-destructive (first-error traceback, no sys.modules surgery) + `_post` retry-once for settings/task POSTs; `build_full.py` reconstructed into the repo (byte-identical rebuild verified; 13 cells ast-validated) | ✅ |
| Oct 8 | **v19 pushed** (kernel version 19) | RUNNING — success = first-error diagnostic visible in install cell (`in-kernel task deps import FAIL: <real cause>`), `TASK SMOKE PASSED` (healthy or clean warn), sync idles on `manifest match` (dataset now populated) |
| Oct 8 | **v19 VERIFIED — all three success criteria met**: (a) install printed the PRECISE first error — subprocess `task deps health: OK` yet `in-kernel task deps import FAIL: ImportError cannot import name '_slice' from 'numpy._core.umath'` → `TASKS_HEALTHY=False`; (b) task smoke upscale `200` REQUIRED + video_frames `200`, bg_remove/extract FAILED (after retry) → `WARN tolerated` → **`TASK SMOKE PASSED`**; (c) sync **`cache dataset up to date (manifest match)`** — write-back idles on the populated v2 dataset, no re-upload | ✅ log `/tmp/opencode/v19out/`. Root-caused the persistence: **subprocess pip swap of numpy in-place** after kernel preload. Locally reproduced 1:1 (2.1.3→2.5.3) + cure proven (force-reinstall to kernel-loaded version) |
| Oct 8 | **v20 built**: install cell heals on EITHER probe (re-pins numpy to the kernel-loaded version, re-verifies both) + **music lane** (core ComfyUI nodes, `_flatten` extraction, `MUSIC_WIDGETS`, preset `music_smoke`, `H3_SMOKE=image,video,music`); rebuild 13 code cells `ast`-OK; **equivalence PASS** (video byte-identical) + music structure checks PASS | ✅ |
| Oct 8 | **v20 pushed** (kernel version 20) — MiniMax Music 3 lane + the proven numpy heal | → v20 run: heal WORKED (`TASKS_HEALTHY=True`), all 3 smokes validated + music generated (AR 57%, tiled decode, lazy switch) but **ERROR at the save step** — see next row |
| Oct 8 | **v20 run root-caused**: `SaveAudioAdvanced.execute() missing 1 required positional argument: 'format'`. Install: `task deps health: OK` (subprocess) / in-kernel FAIL `_slice` → heal `numpy==2.1.3` → **`TASKS_HEALTHY=True`**; image+video smokes ran; **music smoke executed** (`AR sampling: 57% 72/126 [1.17s/it]`, tiled decode node 1008 skipped by lazy switch, 1010 ran) but ERROR at node 35 = SaveAudioAdvanced — `format` field not delivered. **Proven locally** (CPU ComfyUI): `format` is `IO.DynamicCombo`; API prompt needs option-key STRING `"flac"`, v20 sent dict → dropped → missing positional | ❌ run ERROR (but: heal + full music generation verified in production!) |
| Oct 8 | **v21 built+verified**: `_build_music` writes `format` as plain option-key string (+ dotted `format.quality` for mp3/opus); `_flatten` emits dotted dynamic sub-inputs; local CPU ComfyUI: flattened prompt has `format:"flac"`, full prompt validates on the real server with **zero node_errors on SaveAudioAdvanced**, `build_nested_inputs` nests to `{"format":"flac"}` (mp3 → `{"format":"mp3","quality":"V0"}`), v20 dict form reproduces the exact drop | ✅ FIX PROVEN, 13 cells ast-OK (commit `c66cf1f`) |
| Oct 8 | **v21 pushed** (kernel version 21) — music save-node DynamicCombo serialization fix | RUNNING — verify: `TASKS_HEALTHY=True`, `SMOKE PASSED: ['image','video','music']`, real `h3_music_*.flac`, bg_remove/extract green (heal now proven in prod) |

---

## Notes for future agents

- Do NOT hard-code raw tokens in committed files. HF token = Kaggle Secret `HF_TOKEN` first, else
  runtime-assembled fallback; GitHub PAT stays out of the repo entirely.
- All caches/models under `/tmp`, session-only. Dataset mirror via `/kaggle/input` symlinks.
- Dual-T4 designs must use host-staged exchange (`exchange="host"`, `chunks=8`) — PHB, no NVLink.
- Kaggle RAM **31.3 GB** → no pinned weight caches.
- Comfy template JSON = *saved* workflow; `/prompt` = *API* prompt. Always convert; always verify
  against `GET /object_info`; always print the HTTP error **body** (every v5→v12 fix came from it).
- **Verify side effects, don't trust exit codes** (see the no-`__main__`-guard download bug); print
  per-file sizes.
- Reproduce Kaggle failures locally when possible (local CPU ComfyUI dry-run = minutes vs a kernel round trip).
- H3 specifics: no negative prompt; `embedding:<name>` in prompt; duration 5 s min trained; turbo
  = one boolean; template defaults are the int8/nvfp4 quant set.
- **NEVER pop-and-reimport numpy (or any non-trivial extension module) inside the long-lived
  kernel process** — a same-process re-import after `sys.modules` removal is broken by design
  (`cannot load module more than once per process` on multi/ext single-phase init; on Kaggle it
  walks `multiarray._override___module__` and dies with `'numpy.ufunc' object has no attribute
  '__module__'`). "Verify in a fresh interpreter, report in the kernel" — the subprocess probe is
  the heal's executor; the kernel only prints what it sees, never "fixes" itself by reloading C
  extensions (v19).
- **A subprocess pip install can swap numpy files under a running kernel (v17/v19/v20 lesson)**:
  `pip install rembg[cpu]` re-resolves and replaces the numpy wheel in-place. If the kernel
  pre-loaded numpy (torch in preflight), its loaded modules are the OLD version while on-disk
  `strings.py` becomes the NEW version → in-kernel `from scipy import ndimage` / `import rembg`
  dies with `cannot import name '_slice' from 'numpy._core.umath'` while a fresh subprocess probe
  passes on the consistent disk files. Heal = `pip force-reinstall --no-deps numpy==<version THIS
  kernel loaded>` — and **heal when EITHER probe fails** (subprocess-only misses this exact
  divergence). Locally reproduced 1:1 with numpy 2.1.3 → 2.5.3 in-place upgrade (v20 verified the
  heal makes both probes OK again).
- MiniMax Music 3 lane (v20/v21): core ComfyUI nodes only; the music template stores positional
  widgets (no `widgets_values_named`) — promote via the schema-verified `MUSIC_WIDGETS` table;
  music runs single-T4, so the `active: 2 ranks` assertion applies only when the video smoke ran.
- **`IO.DynamicCombo` inputs need the option KEY STRING in the API prompt (v21 lesson)**:
  the saved-workflow widget value is a dict (`{"format":"flac"}`) but ComfyUI's *APIP prompt*
  must carry the plain string `"flac"`; the executor (`get_finalized_class_inputs` →
  `_expand_schema_for_dynamic`) uses it to match the option, inject that option's sub-inputs as
  **dotted keys** (`format.quality`), and `build_nested_inputs` re-nests everything into the dict
  `execute()` receives. A dict value silently fails the match → the input is dropped → a runtime
  `missing positional argument` that `node_errors` validation NEVER catches (validation iterates
  only the expanded schema). Gate such nodes by exercising `get_finalized_class_inputs` +
  `build_nested_inputs` on a local CPU ComfyUI before pushing.
- **Repo build pipeline (Oct 8)**: `build_full.py` IS the source of truth and lives in this repo;
  cells are embedded verbatim as raw `r'''…'''` strings (never regular — cell sources contain
  `\n`/`\.` sequences that regular strings would eat). Rebuild → ast-validate → `kaggle kernels
  push -p .` → commit/push repo + all six MDs. If the host restarts, `/tmp/opencode` dies —
  recover the generator from the committed notebook (reconstruct: read `cells[].source`, join,
  emit raw strings, verify byte-identical against `git show HEAD:…ipynb`).
- After any change: rebuild → `ast`-validate all cells → `kaggle kernels push -p .` →
  `git add -A && git commit && git push` → **update all six MD files**.
