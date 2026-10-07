# Summary / Conclusion

## What we learned on Oct 7 2026

### API lane (`adityahalde8777/minimax-h3-api-server`)

- ✅ WanGP pinned, deps installed, model def discovery fixed (`h3_kaggle_*`).
- ✅ Root cause of HF download failures in WanGP was the patched downloader + HF xet blocked on Kaggle.
- ✅ Fix: disable xet + predownload all assets to `/tmp/Wan2GP/ckpts`.
- ⏳ Single-T4 smoke generation was still running in v6.

### MultiGPU experiment lane (`adityahalde8777/minimax-h3-multigpu-experiment`)

- Host: Kaggle free GPU = **2 × Tesla T4**, PyTorch 2.11 cu128.
- Topology: `GPU0 <-> GPU1 = PHB` (PCIe via host bridge, no NVLink).
- Effective GPU0↔GPU1 exchange: host-staged ≈ **10.0 GB/s**, direct P2P ≈ **9.3 GB/s**.
- ⇒ On T4, the practical dual-GPU mode is **H3 sequence/head split with host-staged exchange** (like Comfy-H3-MultiStream), not raw P2P.
- Multi-GPU single job is the active research direction; throughput two-worker mode is the safe fallback.

## Open questions

1. Does ComfyUI 0.35 + H3 MultiStream mount on Kaggle's image and run one denoise step on 2×T4?
2. WanGP stable lane vs MultiStream lane: which catches the tangent errors first?
3. What torch/deps does the MultiStream lane need to co-exist with WanGP?


## 2026-10-07 final state

- v1 ran end-to-end successfully: smoke test passed, READY block printed, public  URL verified. Total run ≈ 80 min.
- The optional cleanup cell at the bottom ran automatically and stopped the server + tunnel, so v1's endpoint is now dead. The cleanup cell is now guarded by  (v7).
- Only one multi-GPU experiment lane was ever allowed by the 2-session GPU cap; it has been deleted to free a session for the guarded API v7.
