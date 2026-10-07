# MiniMax H3 Kaggle Generator — Comfy 2×T4

Deploy a **MiniMax H3** video/audio generator on Kaggle's free `T4 ×2` machine.
Single lane, single kernel: `adityahalde8777/minimax-h3-comfy-2xt4-generator` (**current build v13**,
**dual-GPU verified**: `active: 2 ranks cuda:0+cuda:1`, both T4s 100 % / ~66 W — quality config
unchanged + full verification suite; see `AGENT.md`).

- **Runtime:** ComfyUI + `ComfyUI-H3-MultiStream` — the H3 transformer split across both T4s
  (`exchange="host"`, `exchange_chunks=8`, caches off for the 31 GB RAM box)
- **Models:** `Comfy-Org/MiniMax-H3` (int8 convrot DiT, NVFP4/AWQ text encoder, int8 video VAE)
- **Workflow:** official `video_minimax_h3_t2v.json`, flattened to API prompt + `H3MultiStream`
  inserted before `BasicGuider`/`BasicScheduler`
- **Quality:** 0.98 MP → **1344×768**, dense (turbo/TeaCache/Spectrum/FBC off)
- **Output:** runs a real smoke generation on both GPUs, then prints a public Base URL +
  `/prompt` endpoint (cloudflared quick tunnel)

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
