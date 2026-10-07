# MiniMax H3 — Kaggle 2×T4 API Server

Deploy a **MiniMax H3 video/audio inference API** on Kaggle free GPUs.

- See [`GUIDE.md`](GUIDE.md) for the full deployment & usage doc.
- See [`AGENT.md`](AGENT.md) for the progressive task/decision log.

TL;DR: push `kernel-metadata.json` + notebook to Kaggle with `kaggle kernels push -p .`,
add an `HF_TOKEN` Kaggle Secret, wait for the READY block, then call the Cloudflare Quick Tunnel URL.

Note on model caching: the default notebook downloads from HF once per kernel. For zero re-downloads, attach a Kaggle Dataset with the H3 files to `/kaggle/input` — the predownload cell now auto-detects and copies them before falling back to HF.
