# MiniMax H3 Kaggle Generator — Comfy 2×T4

Deploy a **MiniMax H3** video/audio generator **+ Flux image generation + use-case tasks** on Kaggle's
free `T4 ×2` machine.
Single lane, single kernel: `adityahalde8777/minimax-h3-comfy-2xt4-generator` (**current build v17 — pushed,
run in progress; v16 VERIFIED**:
dynamic model registry + Kaggle Dataset auto-cache + preset settings + **`/h3api/*` hub API** behind
one public Base URL; **fast smoke verified in v14/v15/v16** = 4-step turbo LoRA @ 864×480, 194–210 s/step,
whole run ~25–33 min, real MP4 out; **hub API + tunnel verified in v15/v16** (all endpoints live, self-tested
through the tunnel); **dataset write-back verified in v16** — `kagglehub upload OK ->
adityahalde8777/minimax-h3-model-cache (530 s)`, 42 GB cached; dual-GPU verified in v13–v16 —
`active: 2 ranks cuda:0+cuda:1`, both T4s 100 % / ~65 W; **v17 local dry-run 59/59 PASSED** —
image lane + 8 use-case tasks verified on CPU before push).

- **Runtime:** ComfyUI + `ComfyUI-H3-MultiStream` — the H3 transformer split across both T4s
  (`exchange="host"`, `exchange_chunks=8`, caches off for the 31 GB RAM box)
- **Image lane (v17):** Flux-schnell fp8 (`Comfy-Org/flux1-schnell`, 17.24 GB → `models/checkpoints/`)
  via the flat `flux_schnell.json` template; `build_prompt()` branches on `modality`; preset
  `image_smoke` (1024², 4 steps, cfg 1, euler/simple); single-T4 (Comfy offload)
- **Use-case tasks (v17):** `POST /h3api/task` — background remover + element extractor (rembg u2net),
  4× upscaler (Real-ESRGAN through Comfy's GPU), video→frames/GIF, audio extract/trim, ffprobe;
  8 tasks, uniform `{ok, outputs, saved, view, meta}` envelope
- **Smoke schedule (v17):** `H3_SMOKE` env (default `image,video`) — one low-quality smoke per
  modality, image first (PNG proves the lane), then video (dual-GPU assertion + MP4);
  per-modality wait budgets (1800 s / 10800 s)
- **Registry:** live HF catalog query every boot (39 core files + flux image tree + task assets +
  100 community repos auto-listed); availability shown as `local` / `cached` / `remote`
- **Cache:** `/kaggle/input` auto-detect → symlink mirror into model dirs → HF fallback (token) →
  `MANIFEST.json`; preset-selective download (only files the ACTIVE preset needs; selecting video
  prunes flux and vice versa)
- **Settings:** `PRESETS` cell — `image_smoke`, `fast_smoke` (turbo ON, 4 steps, 0.4 MP = 864×480)
  and `quality` (dense 20 steps, 0.98 MP = 1344×768); no negative prompt (H3 is CFG-distilled)
- **Workflow:** official `video_minimax_h3_t2v.json`, flattened to API prompt + `H3MultiStream`
  inserted before `BasicGuider`/`BasicScheduler`; shared `build_prompt()` also wired for
  first/last-frame conditioning (data-URL → `LoadImage` → subgraph input)
- **Hub API (v15/v17):** one tunnel → proxy on `:8190` serves the ComfyUI UI, the native `/prompt`
  API **and** `/h3api/*` (docs, health, catalog `?refresh=1`, models, settings, select, download,
  generate, jobs, outputs, free, gpus, import, **tasks**) with WebSocket passthrough; the run
  **self-tests every endpoint live through the tunnel** (retry-once probes, v17) **plus a task
  smoke** before printing READY
- **Write-back (v15/v16 ✅ verified):** downloaded models staged (hardlinks) and pushed back to a
  Kaggle Dataset (`minimax-h3-model-cache`) via **kagglehub's native in-notebook auth** (owner from
  `whoami`, embedded kernel owner as fallback, classic CLI as second fallback) — first run created
  the dataset and pushed 42 GB in 530 s; **200 GB cap auto-skips, never fails the run**
- **Output:** runs a real smoke generation on both GPUs, then prints a public Base URL +
  hub endpoints (cloudflared quick tunnel)

## Quick start

```bash
python3 /tmp/opencode/build_full.py      # regenerate the notebook
kaggle kernels push -p .                 # push the single kernel
kaggle kernels status adityahalde8777/minimax-h3-comfy-2xt4-generator
kaggle kernels logs  adityahalde8777/minimax-h3-comfy-2xt4-generator
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
