# MiniMax H3 Kaggle Generator — Comfy 2×T4

Deploy a **MiniMax H3** video/audio generator on Kaggle's free `T4 ×2` machine.
Single lane, single kernel: `adityahalde8777/minimax-h3-comfy-2xt4-generator` (**current build v15**:
dynamic model registry + Kaggle Dataset auto-cache + preset settings + **`/h3api/*` hub API** behind
one public Base URL; **fast smoke verified in v14** = 4-step turbo LoRA @ 864×480, 210 s/step,
whole run 25.6 min, real MP4 out; dual-GPU verified in v13/v14 — `active: 2 ranks cuda:0+cuda:1`,
both T4s 100 % / ~65 W).

- **Runtime:** ComfyUI + `ComfyUI-H3-MultiStream` — the H3 transformer split across both T4s
  (`exchange="host"`, `exchange_chunks=8`, caches off for the 31 GB RAM box)
- **Registry:** live HF catalog query every boot (39 core files + 100 community repos auto-listed);
  availability shown as `local` / `cached` / `remote`
- **Cache:** `/kaggle/input` auto-detect → symlink mirror into model dirs → HF fallback (token) →
  `MANIFEST.json`; preset-selective download (only files the ACTIVE preset needs)
- **Settings:** `PRESETS` cell — `fast_smoke` (turbo ON, 4 steps, 0.4 MP = 864×480) and `quality`
  (dense 20 steps, 0.98 MP = 1344×768); no negative prompt (H3 is CFG-distilled)
- **Workflow:** official `video_minimax_h3_t2v.json`, flattened to API prompt + `H3MultiStream`
  inserted before `BasicGuider`/`BasicScheduler`; shared `build_prompt()` also wired for
  first/last-frame conditioning (data-URL → `LoadImage` → subgraph input)
- **Hub API (v15):** one tunnel → proxy on `:8190` serves the ComfyUI UI, the native `/prompt`
  API **and** `/h3api/*` (docs, health, catalog `?refresh=1`, models, settings, select, download,
  generate, jobs, outputs, free, gpus, import) with WebSocket passthrough
- **Write-back (v15):** downloaded models staged (hardlinks) and pushed back to a Kaggle Dataset
  (`MiniMax H3 model cache`) — **200 GB cap auto-skips, never fails the run**
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
