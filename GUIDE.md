# MiniMax H3 → Kaggle 2×T4 API Server — Deployment Guide

This repo packages a **headless MiniMax H3 video/audio inference API** designed to run on the Kaggle
free tier (`NvidiaTeslaT4` = **2× T4, 29 GB RAM, PCIe**).

Current stable status: **single-GPU WanGP path works**; **dual-T4 H3 MultiStream path is the active
experimental direction**.

## Contents

| Path | Purpose |
|---|---|
| `minimax_h3_api.ipynb` | Kaggle notebook — full deployable server |
| `kernel-metadata.json` | Kaggle CLI metadata (GPU, internet, machine shape) |
| `assets/h3_server.py` | FastAPI inference server (source of truth) |
| `assets/h3_cli.py` | Local CLI client for the API |
| `AGENT.md` | Working protocol + task/decision log for contributors |

## Quick deploy to Kaggle

```bash
# 1. Configure Kaggle creds
mkdir -p ~/.kaggle && cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"<YOU>","key":"<KEY>"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# 2. Push the kernel (creates + runs it)
cd /path/to/this/repo
kaggle kernels push -p .
```

Then add `HF_TOKEN` in Kaggle → your kernel → **Add-ons → Secrets** (strongly recommended even though
the repo is public; it avoids rate limits). Everything heavy (models, caches) lives under `/tmp`.

## API contract

```bash
export H3_BASE_URL="https://<random>.trycloudflare.com"
export H3_API_KEY="<printed in the notebook READY block>"

curl "$H3_BASE_URL/health"

curl -X POST "$H3_BASE_URL/v1/videos" \
  -H "Authorization: Bearer $H3_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"fl2va","prompt":"A cinematic ...","resolution":"1344x768","duration_seconds":5,"steps":28,"quality":"maximum"}'

curl "$H3_BASE_URL/v1/videos/<JOB_ID>" -H "Authorization: Bearer $H3_API_KEY"
curl -L "$H3_BASE_URL/v1/videos/<JOB_ID>/download" -H "Authorization: Bearer $H3_API_KEY" -o out.mp4
```

## Architecture (current)

```
Kaggle T4×2 machine
 ├── WanGP (pinned commit) → FastAPI server on 127.0.0.1:8000
 ├── cloudflared Quick Tunnel → public https://*.trycloudflare.com
 ├── H3 FL2VA INT8 ConvRot + Qwen3-VL 32B Q4_K_M
 └── /tmp — models, checkpoints, caches, outputs (session-only)
```

## Why T4×2 matters

Earlier we threw away the second T4. `NvidiaTeslaT4` on Kaggle is **T4×2**, and the H3 MultiStream
project shows real single-job multi-GPU H3 acceleration. See `AGENT.md` for the active architecture
scratchpad and the dual-T4 design that is being built next.

## Quality defaults

- `1344×768` (~1 MP, H3 16:9 canvas), 28 steps, Euler, SDPA, Profile 4, `vmm_spill`
- No turbo LoRA, no TeaCache, no Spectrum, no first-block cache

## Known issues / roadmap

Tracked in `AGENT.md`.
