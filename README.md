# MiniMax H3 Kaggle Generator — Comfy 2×T4

Deploy a **MiniMax H3** video/audio generator **+ Flux image generation + MiniMax Music 3 music
generation + use-case tasks** on Kaggle's
free `T4 ×2` machine.
Single lane, single kernel: `adityahalde8777/minimax-h3-comfy-2xt4-generator` (**current build v21 —
pushed; v20 ran the full pipeline but ERRORed at the music save step → root-caused + fixed in v21**;
v19 ran COMPLETE — v18/v17/v16 verified — see below**:
dynamic model registry + Kaggle Dataset auto-cache + preset settings + **`/h3api/*` hub API** behind
one public Base URL; **fast smoke verified in v14–v18** = 4-step turbo LoRA @ 864×480, 194–210 s/step,
whole run ~25–33 min, real MP4 out; **hub API + tunnel verified in v15–v18** (all endpoints live, self-tested
through the tunnel); **dataset write-back verified in v16+v18** — `kagglehub upload OK ->
adityahalde8777/minimax-h3-model-cache` (v16: 530 s create, v18: 716 s version 2), 59.3 GB cached;
dual-GPU verified in v13–v18 — `active: 2 ranks cuda:0+cuda:1`, both T4s 100 % / ~65 W; **v17/v18
proved the multi-modality lanes on Kaggle** — image smoke PNG + video MP4 @ 204–206 s/step × 4 in one
run, pub self-test green, upscale + video_frames tasks `200`, `TASK SMOKE PASSED` in v18 with sync
running; **v19 verified the non-destructive first-error diagnostic** (`in-kernel task deps import
FAIL: ImportError cannot import name '_slice' from 'numpy._core.umath'` — root-caused as a
subprocess pip install swapping numpy **in-place** after the kernel pre-loaded it) and the sync
cell **idled on `manifest match`** (write-back done); **v20 PROVED the numpy heal in production**
(`TASKS_HEALTHY=True` after re-pinning numpy to the kernel-loaded version) and ran the **music
lane end-to-end** — AR sampling to 57 %, tiled decode, lazy switch — but ERRORed at the last op:
`SaveAudioAdvanced.execute() missing 'format'`; **v21 FIXED that** (DynamicCombo option-key must be
a plain string `"flac"` in the API prompt; a dict silently drops the input) — root cause proven on
a local CPU ComfyUI before the push; **`build_full.py` now lives in this repo** (survives host restarts)).

- **Runtime:** ComfyUI + `ComfyUI-H3-MultiStream` — the H3 transformer split across both T4s
  (`exchange="host"`, `exchange_chunks=8`, caches off for the 31 GB RAM box)
- **Image lane (v17):** Flux-schnell fp8 (`Comfy-Org/flux1-schnell`, 17.24 GB → `models/checkpoints/`)
  via the flat `flux_schnell.json` template; `build_prompt()` branches on `modality`; preset
  `image_smoke` (1024², 4 steps, cfg 1, euler/simple); single-T4 (Comfy offload)
- **Music lane (v20/v21):** MiniMax Music 3 — **ComfyUI core nodes only** (`nodes_minimax_music.py`,
  no custom pack, no MultiStream, single T4); `audio_minimax_music_3.json` subgraph flattened by
  the shared `_flatten(wf, info)` (extracted from the video path with byte-identical output);
  preset `music_smoke` (30 steps cfg 1.7, max_duration 5, seed 4242, tiled decode, `flac` →
  `h3_music_*.flac`); 14.33 GB models from `Comfy-Org/MiniMax-Music-3`; **v21**: SaveAudioAdvanced
  `format` (an `IO.DynamicCombo`) is sent as the plain option-key string `"flac"` in the API
  prompt (mp3/opus add the dotted `format.quality` sub-input) — the dict form silently drops it
- **Use-case tasks (v17):** `POST /h3api/task` — background remover + element extractor (rembg u2net),
  4× upscaler (Real-ESRGAN through Comfy's GPU), video→frames/GIF, audio extract/trim, ffprobe;
  8 tasks, uniform `{ok, outputs, saved, view, meta}` envelope
- **Smoke schedule (v17/v20):** `H3_SMOKE` env (default `image,video,music`) — one low-quality
  smoke per modality, image first (PNG proves the lane), then video (dual-GPU assertion + MP4),
  then music (audio `.flac` assertion); per-modality wait budgets (image 1800 s / video 10800 s /
  music 3600 s)
- **Registry:** live HF catalog query every boot (H3 core 39 files + flux image tree + **Music-3
  tree** + task assets + 100 community repos auto-listed); availability shown as `local` /
  `cached` / `remote`
- **Cache:** `/kaggle/input` auto-detect → symlink mirror into model dirs → HF fallback (token) →
  `MANIFEST.json`; preset-selective download (only files each modality's preset needs; selecting
  video prunes flux and vice versa); **write-back verified twice and now idles on `manifest
  match`** (v19)
- **Settings:** `PRESETS` cell — `image_smoke`, `fast_smoke` (turbo ON, 4 steps, 0.4 MP = 864×480),
  `quality` (dense 20 steps, 0.98 MP = 1344×768) and **`music_smoke`** (30 steps, 5 s, flac); no
  negative prompt (H3 is CFG-distilled)
- **Workflow:** official `video_minimax_h3_t2v.json`, flattened to API prompt + `H3MultiStream`
  inserted before `BasicGuider`/`BasicScheduler`; shared `build_prompt()` also wired for
  first/last-frame conditioning (data-URL → `LoadImage` → subgraph input)
- **Hub API (v15/v19):** one tunnel → proxy on `:8190` serves the ComfyUI UI, the native `/prompt`
  API **and** `/h3api/*` (docs, health, catalog `?refresh=1`, models, settings, select, download,
  generate, jobs, outputs, free, gpus, import, **tasks**) with WebSocket passthrough; the run
  **self-tests every endpoint live through the tunnel** (retry-once probes, v17; POSTs got the
  same retry-once treatment in v19) **plus a task smoke** before printing READY
- **Write-back (v15/v16/v18 ✅ verified):** downloaded models staged (hardlinks) and pushed back to a
  Kaggle Dataset (`minimax-h3-model-cache`) via **kagglehub's native in-notebook auth** (owner from
  `whoami`, embedded kernel owner as fallback, classic CLI as second fallback) — v16 created the
  dataset (42 GB, 530 s), **v18 pushed version 2** (716 s, manifest drift 5→6) after the
  `TASKS_HEALTHY` gate let a degraded-but-healthy run finish; **200 GB cap auto-skips, never fails
  the run**
- **Resilience (v18/v19/v20/v21):** install cell probes the exact CPU-task imports in a subprocess
  AND in the kernel, and auto-heals numpy with a same-version `force-reinstall` **whenever either
  probe fails** (proven root cause: a subprocess `pip install` swaps numpy files in-place after
  the kernel pre-loaded numpy — the disk looks fine to a fresh interpreter while in-kernel imports
  break with `_slice`); the heal was **PROVEN in production by v20** (`TASKS_HEALTHY=True`);
  in-kernel verify is **non-destructive** and prints the real first error; `TASKS_HEALTHY` gates
  bg_remove/extract to warn-only when the environment is broken (upscale stays REQUIRED) so **the
  run always completes**; pub POSTs retry once; **v21 added a lesson the hard way**: `IO.DynamicCombo`
  inputs in the API prompt are option-key strings, not widget dicts
- **Output:** runs a real smoke generation on both GPUs, then prints a public Base URL +
  hub endpoints (cloudflared quick tunnel)

## Quick start

```bash
python3 build_full.py                    # regenerate the notebook (build script lives in this repo)
python3 local_check.py                   # pre-push gate: ast + build 3 lanes vs fixture (no deps)
# full gate (recommended before a push that touches convert/_flatten):
#   /path/to/ComfyUI/venv/bin/python local_check.py --live http://127.0.0.1:8191 --comfy /path/to/ComfyUI
kaggle kernels push -p .                 # push the single kernel
kaggle kernels status adityahalde8777/minimax-h3-comfy-2xt4-generator
kaggle kernels outputs adityahalde8777/minimax-h3-comfy-2xt4-generator -p <dir>   # log after COMPLETE
```

## Docs (update ALL of them after any significant change)

| File | Purpose |
|---|---|
| `README.md` | this at-a-glance page |
| `GUIDE.md` | step-by-step run + cell-by-cell map |
| `BUILD_YOUR_OWN.md` | reproduce the same stack for your own model |
| `AGENT.md` | mandatory task + decision log |
| `CONCLUSION.md` | findings, hard numbers, open questions |
| `ERROR_PLAYBOOK.md` | every failure chain (downloads + `/prompt`) and its fix |
