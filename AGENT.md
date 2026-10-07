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

**Date:** Oct 7 2026
**Owner:** adityahalde8777
**Kaggle kernel:** `adityahalde8777/minimax-h3-comfy-2xt4-generator` (the ONLY kernel; old lanes deleted)
**Latest:** **v14 PUSHED** (dry-run green locally): dynamic registry + Dataset auto-cache +
preset settings (`fast_smoke` = 4-step turbo @ 864×480), `/prompt` accepted offline
(`prompt_id 918c50eb…`). **v13 still RUNNING** (dual-GPU verified live: `active: 2 ranks
cuda:0+cuda:1`, both T4s 100 %/66 W). Next: watch v14's fast smoke → READY block.

---

## Mission (expanded scope — user directives, Oct 7)

The product is an **all-in-one generation hub** on one public Base URL, not just an H3 video demo:

| Modality | Models (candidates) | Status |
|---|---|---|
| **Video generation** | MiniMax H3 (FL2VA/Ref2VA, turbo/dense), H3-Max-style variants | ✅ H3 lane in production |
| **Image generation** | Flux family (Comfy-Org repos), community Fluxes | ⬜ Phase C |
| **Music generation** | `audio_minimax_music_3` Comfy template (MinimaxMusic), LTx community | ⬜ Phase C |
| **Sound / soundtrack** | H3 native audio track (already joint audio+video), music models | ⬜ Phase C |
| **Transcription (ASR)** | to research (Whisper-class on Comfy / community) | ⬜ Phase C research |
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
| `/tmp/opencode/build_full.py` | **Notebook generator** (edit → `python3 build_full.py` writes the `.ipynb`). MUST be moved into this repo (Phase E) |
| `/home/limitlessjourney829/kaggle/minimax-h3-api/minimax_h3_full_comfy.ipynb` | generated notebook (pushed with `kaggle kernels push -p .`) |
| `/home/limitlessjourney829/kaggle/minimax-h3-api/kernel-metadata.json` | kernel metadata (private, GPU, internet, T4) |
| `/tmp/opencode/h3_t2v.json` | official template `video_minimax_h3_t2v.json` (local copy, converter dry-runs) |
| `/tmp/opencode/ComfyUI/` | local ComfyUI clone (CPU dry-runs; needs pip CPU install) |
| `/tmp/opencode/Comfy-H3-MultiStream/` | node-pack clone (docs, perf, split.py log lines) |
| Repo | https://github.com/adittaya/kaggle-model-notebook-implementation-guide (public) |

Notebook cell order (v13, 10 cells): title → secrets → preflight → install → assets → inspect →
start → smoke (converter+submit) → wait_smoke (verify) → pub (cloudflared tunnel → READY).

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

### Phase A — v14: dynamic foundation + fast verified smoke (PUSHED — watching)
1. ✅ **Registry cell**: live HF tree + community search → grouped catalog with
   `local/cached/remote` status — new files/repos auto-list every boot
2. ✅ **Settings cell**: `PRESETS` + `ACTIVE` (`fast_smoke`: turbo ON, 4-step lora, 0.4 MP =
   864×480, duration 5, short prompt, seed) — single source of truth for prompt + downloads
3. ✅ **Assets rework**: preset-selective download, local → `/kaggle/input` symlink mirror → HF
   (token), `MANIFEST.json`, per-file `OK <bytes>`
4. ✅ Converter applies `ACTIVE` (instance widgets + ResolutionSelector); wait cap → 3 h;
   dual-GPU verification retained
5. ✅ Local CPU-Comfy dry-run of the turbo prompt → **`/prompt` accepted** → pushed v14
6. ⬜ Watch v14 smoke → capture READY block (Base URL + `/prompt` endpoint) + step timings

### Phase B — enterprise hybrid API (v15)
1. ⬜ Sidecar (`/h3api/*`) + reverse proxy (WebSocket passthrough for Comfy UI) behind the ONE tunnel
2. ⬜ Endpoints: `/h3api/catalog`, `/models`, `/settings` GET/POST, `/download`, `/select`
   (download-on-demand → Comfy `/free` unload → optional purge = clean space/GPU),
   `/generate`, `/jobs`, `/outputs`, `/free`, `/import` (custom repo download = "upload"),
   `/gpus` (live nvidia-smi + MultiStream evidence)
3. ⬜ World-class JSON: consistent envelope (ok/error/code/message), job objects, presets echo
4. ⬜ Docs: `GET /h3api/docs` (served markdown/OpenAPI) + repo `API.md` — every endpoint + helper
5. ⬜ Kaggle Dataset write-back with **200 GB cap guard** (auto-skip when full, log + continue)

### Phase C — modalities (v16+, one low-quality smoke each, in order)
1. ⬜ **Image generation**: locate Flux/Comfy-Org image repos + official API template → lane → smoke
2. ⬜ **Music generation**: `audio_minimax_music_3.json` template (+ LTx/community research) → smoke
3. ⬜ **Transcription**: research ASR options on Comfy → lane → smoke
4. ⬜ **Soundtrack**: H3 native joint audio (already produces 32 kHz stereo) — document + API field
5. ⬜ Each modality gets: registry entries, settings schema, docs section, one 480p/low smoke

### Phase D — community model integration
1. ⬜ Ref2VA video lane (`api_minimax_h3_r2v.json` + ref DiT + ref2v turbo LoRA) → smoke
2. ⬜ First/last-frame I2V via API (image upload → `first_frame`/`last_frame`)
3. ⬜ Community imports: lightx2v v1.1/v1.2 turbos, style LoRAs, ControlNet, detail VAE, upscaler
4. ⬜ `/h3api/import {repo, files}` for arbitrary models; catalog auto-grows

### Phase E — housekeeping
1. ⬜ Move `build_full.py` into this repo (source of truth; token injected from outside git)
2. ⬜ v13 outcome: capture step-time diagnostics (`[MultiStream] step:`), READY block; stop if stale
3. ⬜ Perf: explain 861 s/step; levers = turbo (done in A), 0.4 MP, later VAE-split + TE cache
4. ⬜ Dataset cache bootstrap: first write-back populates the 200 GB dataset → later boots = instant
5. ⬜ Rotate the GitHub PAT pasted in chat earlier; keep HF token out of the public repo

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
- After any change: rebuild → `ast`-validate all cells → `kaggle kernels push -p .` →
  `git add -A && git commit && git push` → **update all six MD files**.
