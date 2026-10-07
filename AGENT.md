# AGENT.md — Progressive Work, Task Tracking, Decisions

**Rule:** Every meaningful change must add one line to *Task Log* and one line to *Decision Log*.
Keep the current kernel status up to date at the top.

**Date:** Oct 7 2026
**Owner:** adityahalde8777
**Kaggle kernel:** `adityahalde8777/minimax-h3-api-server`

---

## Current status

| Area | Status |
|---|---|
| Notebook | `minimax_h3_api.ipynb` pushed (v6) |
| MultiGPU experiment | v2 running |
| Model predownload fix | ✅ `HF_HUB_DISABLE_XET=1`, assets staged to `/tmp/Wan2GP/ckpts` |
| WanGP single-T4 worker | ✅ stable, smoke test in progress on last run |
| Cloudflare tunnel + FastAPI | ✅ working when single-T4 path passes smoke test |
| Dual-T4 single-job H3 | 🚧 experimental lane, not yet wired |
| H3 MultiStream (ComfyUI) | ⬜ research stage, benchmark numbers validated but not implemented |

---

## To do

1. ⬜ Verify v6 single-T4 notebook finishes smoke test → prints READY block
2. ⬜ Add `machine_shape: NvidiaTeslaT4` to `kernel-metadata.json` to guarantee T4×2 allocation on push
3. ⬜ Add GPU probe cell (`torch.cuda.device_count()`, `nvidia-smi topo -m`, free VRAM/RAM) logging
4. ⬜ Add `gpu_mode: "single" | "dual" | "throughput"` to API payload + `/v1/system`
5. ⬜ Implement throughput mode: 2 independent WanGP workers (`CUDA_VISIBLE_DEVICES=0/1`) behind one scheduler
6. ⬜ Evaluate H3 MultiStream (ComfyUI 0.35+) on T4×2 with `exchange=host`, `exchange_chunks=8`
7. ⬜ Only if MultiStream is stable: switch default `gpu_mode="dual"`, keep WanGP as fallback

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| Oct 7 | Disable HF xet downloads (`HF_HUB_DISABLE_XET=1`) | Kaggle blocks/throttle xet; asset downloads failed with "cannot find on Hub" |
| Oct 7 | Pre-download assets via `hf_hub_download(local_dir=...)` | WanGP patched HTTP downloader masked errors and failed on the pinned revision path |
| Oct 7 | Use `/resolve/main/` URLs in finetune defs | Sha-pinned URLs triggered WanGP path with no useful diagnostics |
| Oct 7 | Match pinned model type by `model_type` prefix `h3_kaggle_` | Name match on "Kaggle" failed; types now discovered |
| Oct 7 | T4 2nd GPU left as fallback to WanGP single on 1 T4 | Multi-GPU H3 via WanGP is not the stable line; Comfy MultiStream is the experimental lane |
| Oct 7 | Keep H3 1344×768 ~1 MP as default | User specified; avoids 2× token cost of 1080p on T4 |
| Oct 7 | H3-Base locally; Context-IR/Regenerate-2K are hosted APIs | MiniMax docs state they are not open-sourced |

---

## Task log

| Date | Task | Result |
|---|---|---|
| Oct 7 07:34 | Initial Kaggle workspace, created project skeleton | Done |
| Oct 7 07:48 | v1 push | Failed: model discovery `RuntimeError: pinned FL2VA not found` |
| Oct 7 | v2 patch of name→model_type match | Progressed; smoke test error masked by non-str GenerationError list |
| Oct 7 | v3 error rendering patch | Surfaced: patched HTTP downloader broke |
| Oct 7 | v4 switched to `/resolve/main/` URLs | Surfaced: HF xet unable to locate on Hub |
| Oct 7 | v5 set `HF_HUB_DISABLE_XET` | Still failed in WanGP downloader path |
| Oct 7 | v6 pre-download assets in notebook | ✅ All 12 assets downloaded; smoke test entered; final validation pending |
| Oct 7 | MultiGPU experiment v1 | probe finished: 2× T4 cc7.5 14.6G, PHB topology, host~10.0 GB/s vs p2p~9.3 GB/s; failed only on cell syntax typo |
| Oct 7 | MultiGPU experiment v2 | probe finished green: host~10.3 GB/s vs p2p~9.4 GB/s; PHB topology. v3 SUCCESS on rerun. |

---

## Notes for future agents

- Do NOT hard-code tokens in the notebook. Use Kaggle Secrets (`HF_TOKEN`, `KAGGLE_KEY` env vars for fallback).
- All caches/models under `/tmp`, session-only.
- Dual-T4 designs must use host-staged exchange, not raw P2P, on T4 PCIe.
- Turbo/TeaCache/Spectrum/FBC stay OFF for the quality-first build.

| Oct 7 | Notebook predownload now prefers /kaggle/input datasets; documented in GUIDE/BUILD_YOUR_OWN | ✅ |
| Oct 7 | v1 full run: READY block printed, URL/key valid, ~80 min; cleanup cell killed server → cleanup now guarded by H3_SHUTDOWN=1 | ✅ |
| Oct 7 | v7 pushed (guarded cleanup + dataset-aware predownload) as the primary API path | RUNNING |
| Oct 7 | v8 prepared: predownload cell now writes MANIFEST.json + mirrors /kaggle/input first | NEXT PUSH |
| Oct 7 | cleaned all nexus/ltx/probe/notebook kernels; only minimax-h3-kaggle-generator remains | ✅ |
| Oct 7 | Added ERROR_PLAYBOOK.md documenting the xet-download / patched-downloader failure chain and the canonical fix | ✅ |
| Oct 7 | Rebuilt primary generator notebook with T4x2 probe cell; WanGP single-T4 remains stable fallback; dual H3 MultiStream lane is experimental next | ✅ |
| Oct 7 | pushed a separate comfy 2xT4 kernel (ComfyUI + H3 MultiStream) so we can smoke on both T4s | BUILDING |
