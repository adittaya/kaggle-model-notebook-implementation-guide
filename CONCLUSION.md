# Summary / Conclusion

## 2026-10-07 — Comfy 2×T4 lane (current, single lane)

### Architecture finally chosen
- **One kernel only:** `adityahalde8777/minimax-h3-comfy-2xt4-generator`.
- ComfyUI (Comfy-Org/ComfyUI @ main) + `ComfyUI-H3-MultiStream` node pack, models from
  `Comfy-Org/MiniMax-H3`, official `video_minimax_h3_t2v.json` template.
- Public endpoint: Comfy's own `/prompt` API behind a cloudflared quick tunnel, printed only
  after a real smoke generation completes on **both** T4s — and "both" is now **enforced**:
  a background `nvidia-smi -l 10` monitor feeds `[gpu]` utilization lines into the wait loop,
  and the wait cell raises unless comfy.log contains `[MultiStream] active: 2 ranks`
  (single-GPU `UNSPLIT` fallback = hard failure).

### What the debugging marathon (v3 → v11) taught us

| # | Failure | Real lesson |
|---|---|---|
| v3 | `ComfyUI did not listen in time` | Never `pip install -U huggingface_hub` — it drags in 2.x and breaks ComfyUI's pinned gradio/tokenizers stack |
| v5 | `HTTP 500` on `/prompt` | Saved workflow JSON (`nodes`/`links`) ≠ API prompt (`{id:{class_type,inputs}}`) |
| v8 | `missing_node_type: MarkdownNote` | UI-only nodes exist in the graph but not in `object_info`; filter them out |
| v9 | `json.load(bytes)` | One-shot typo; the value of printing the response **body** |
| v10 | `KeyError` validating `SaveVideo` | The whole H3 pipeline lives inside `definitions.subgraphs[0]` — skipping an unresolvable node leaves dangling links; you must **flatten** subgraphs, not drop them |
| v11 | 6 × `value_not_in_list` + `values.a` missing | The asset cell had **never downloaded anything**: `python -m huggingface_hub.cli.download` has no `__main__` guard (exit 0, silence). Also: static `object_info` can omit dynamically finalized inputs (`values.a`), so **links must always be passed through** |
| v12 | *perf anomaly:* **861 s/step** measured on 2×T4 (≈4.8 h for 20 steps) — ~17× slower than the pack's 2×5060 Ti reference (42–49 s/step); its 60-min wait cap would have false-failed a healthy run | size every timeout from **measured** throughput and stream the pack's per-step diagnostics live; verify the split is active *before* trusting the timing |
| (v13) | *prevention, not a failure* | Multi-GPU intent in a submitted config is not proof of execution: the split can silently fall back to one card. Assert the pack's runtime evidence (`active: 2 ranks` vs `UNSPLIT`) and tail `nvidia-smi` into the output |
| v12 | (pending) | `snapshot_download` + size verification, link inputs always kept, flattener otherwise unchanged |

### Hard numbers measured on the Kaggle image
- 2 × Tesla T4 (14.6 GB each), topology `PHB`, **host ≈ 10 GB/s vs p2p ≈ 9.3 GB/s**
  ⇒ `exchange="host"`, `exchange_chunks=8`.
- **System RAM = 31.3 GB** ⇒ all pinned weight caches OFF (node-pack README: 32 GB class ⇒
  caches disabled; the DiT cache alone wants 18 GiB).
- Resolution table from the template: **0.98 MP = 1344×768** (official 768p) — the target quality row.

## 2026-10-07 — retired lanes (kept for history)

### API lane (`minimax-h3-api-server`, deleted)
- ✅ WanGP pinned, deps installed, model-def discovery fixed (`h3_kaggle_*`).
- ✅ Root cause of HF download failures: patched WanGP downloader + HF xet blocked on Kaggle.
- ✅ Fix: `HF_HUB_DISABLE_XET=1` + predownload all assets to `/tmp/Wan2GP/ckpts`.
- ✅ v1 ran end-to-end, READY block + public URL, ≈ 80 min; cleanup cell used to kill the
  endpoint immediately → now guarded by `H3_SHUTDOWN=1`.

### MultiGPU experiment lane (deleted)
- Effective exchange host ≈ 10.0 GB/s vs p2p ≈ 9.3 GB/s ⇒ host-staged split, not raw P2P.
- This is what made us choose Comfy-H3-MultiStream's `exchange="host"` default.

## Open questions

1. ✅ Assets land (v12: 5/5 `OK <bytes>`, 41 GB) and the flattened prompt validates
   (`prompt_id … node_errors: {}`) — the pipeline works end to end.
2. ✅ **20-step dense 1344×768 on 2×T4 ≈ 861 s/step → ≈4.8 h sampling** (v12, first measurement).
   Why ~17× slower than the pack's 2×5060 Ti reference (42–49 s/step)? v13 streams
   `[MultiStream] step:` / `[GPUs]` diagnostics to separate compute vs exchange.
3. ✅ **Is the split actually on both T4s?** — **YES, verified live in v13**:
   `[GPUs] dit: 2 rank(s): cuda:0 (primary), cuda:1` + `active: 2 ranks cuda:0+cuda:1,
   50 blocks, heads 28/28` + both cards at `100 % util / ~66 W (T4 ceiling)`,
   10.4 / 8.6 GiB resident; no `UNSPLIT`. The final assertion confirms it at smoke end.
4. Does `weight_cache=False` leave enough throughput, or do we need a Dataset-backed boot to
   afford the 18 GiB pinned DiT cache?
5. Optional speed lever (quality kept for now): the template's `turbo_mode` switch → 4-step
   turbo lora path; also `H3 MS VAE Split Decode` + `H3 MS Text Encoder Cache`.
