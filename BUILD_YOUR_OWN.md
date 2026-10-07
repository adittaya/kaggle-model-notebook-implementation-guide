# Build Your Own — Same Technology & Features

This document is the checklist for making your **own** model-server notebook on Kaggle that reproduces the same style of features as this repo.

## The stack

```
Your model repo on HF
        │
        ▼
Kaggle kernel (T4×2, internet on)
        │
        ├─ WanGP pinned to a commit  ← stable generative runtime
        ├─ Local pinned model definitions (finetunes/*.json)
        ├─ FastAPI headless server (no Gradio)
        ├─ Cloudflare Quick Tunnel → public https://*.trycloudflare.com
        ├─ All caches/models in /tmp (session-only)
        └─ Polling job API (works on Quick Tunnels; SSE doesn't)
```

## Feature checklist

- [ ] **Pinned upstream**: clone WanGP (or your runtime) and `git checkout <commit>`
- [ ] **Pinned assets**: per-H3 finetune JSON pointing to exact HF repos/revisions
- [ ] **Auth from secrets**: never hardcode tokens; read `HF_TOKEN`, `KAGGLE_KEY` from Kaggle Secrets / env
- [ ] **Cache hygiene**: `HF_HOME=/tmp/hf`, `TORCH_HOME=/tmp/torch`, `XDG_CACHE_HOME=/tmp/xdg_cache`
- [ ] **Predownload**: download all assets with `hf_hub_download(local_dir=...)` before first generation
- [ ] **Disable broken downloaders**: `HF_HUB_DISABLE_XET=1`, `HF_HUB_DISABLE_HF_TRANSFER=1`
- [ ] **Headless API**: FastAPI with `Authorization: Bearer`, async job + polling, `/health`, `/v1/system`, `/v1/videos/{id}`
- [ ] **Smoke test before READY**: run a small real generation; only then print BASE URL + API KEY
- [ ] **Quality-first defaults**: no turbo loras, no TeaCache/Spectrum/FBC, SDPA attention, vmm spill
- [ ] **Public tunnel**: `cloudflared tunnel --url http://127.0.0.1:8000`
- [ ] **Graceful cleanup cell** at the bottom

## 2×T4 T4×2 guide (Kaggle)

Kaggle `NvidiaTeslaT4` = 2× T4, 29 GB RAM, PCIe, no NVLink.

- Topology: `nvidia-smi topo -m` → `GPU0 ↔ GPU1 = PHB`
- Effective exchange: host ≈ 10 GB/s, p2p ≈ 9.3 GB/s ⇒ use **`exchange="host"`** + `exchange_chunks=8`
- for real single-job multi-GPU H3, use the H3 MultiStream lane, not WanGP mainline
- throughput mode (2 independent WanGP workers, one per GPU) is the low-risk fallback

## Your repository structure

Copy this shape:

```
your-kernel-repo/
├── README.md                  # quick intro
├── GUIDE.md                   # deployment guide
├── BUILD_YOUR_OWN.md          # this file = the "same features" source
├── AGENT.md                   # every task/decision line MUST be logged
├── CONCLUSION.md              # what we learned and why
├── kernel-metadata.json       # machine_shape: NvidiaTeslaT4, internet on
├── your_notebook.ipynb
└── assets/
    ├── server.py              # FastAPI app
    └── cli.py                 # local client
```

## Mandatory dev discipline

1. Every run → append to `AGENT.md` task log + decision log
2. Every real configuration result → append to `CONCLUSION.md`
3. `git add -A && git commit -m "..." && git push` after every significant change
4. Don't modify v1 in-place after it works; create v2 sideways like we did
5. Never commit credentials; rotate any leaked PAT/token

## Minimal viable command set

```bash
kaggle kernels push -p .
kaggle kernels status OWNER/kernel
kaggle kernels logs OWNER/kernel
kaggle kernels delete -y OWNER/kernel/old

git add -A; git commit -m "progress: ..."; git push
```

Use this template and you're ~90% of the way to the same MiniMax H3 API server we built.

---

## Persistent model caching across sessions (Kaggle Dataset)

If you don't want to re-download ~20 GB of weights every new kernel, ship them in a Kaggle Dataset and let the notebook **auto-detect** the local mount:

1. Upload the assets once: `kaggle datasets create -p <folder with the H3 files>`
2. In your notebook's predownload cell, **first** mirror anything under `/kaggle/input/` into `<ckpts>/`:

```python
CKPT = Path("/tmp/Wan2GP/ckpts"); CKPT.mkdir(parents=True, exist_ok=True)
for base in [Path("/kaggle/input")]:
    for root,_,files in os.walk(base):
        for f in files:
            if "minimax-h3" in f.lower() or "qwen3-vl" in f.lower() or "minimax_h3" in f.lower():
                dst = CKPT / ("Qwen3-VL-32B-Instruct" if "qwen3-vl" in f.lower() else ("minimax_h3" if f.startswith("minimax_h3") else "")) / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(os.path.join(root,f), dst)
```

3. Then only run `hf_hub_download` for files **still missing** from `CKPT`.
