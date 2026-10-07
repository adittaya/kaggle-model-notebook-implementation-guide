# Summary / Conclusion

## 2026-10-07 — Comfy 2×T4 lane (current, single lane)

### Architecture finally chosen
- **One kernel only:** `adityahalde8777/minimax-h3-comfy-2xt4-generator`.
- ComfyUI (Comfy-Org/ComfyUI @ main) + `ComfyUI-H3-MultiStream` node pack, models from
  `Comfy-Org/MiniMax-H3`, official `video_minimax_h3_t2v.json` template.
- Public endpoint: Comfy's own `/prompt` API behind a cloudflared quick tunnel, printed only
  after a real smoke generation completes on **both** T4s.

### What the debugging marathon (v3 → v11) taught us

| # | Failure | Real lesson |
|---|---|---|
| v3 | `ComfyUI did not listen in time` | Never `pip install -U huggingface_hub` — it drags in 2.x and breaks ComfyUI's pinned gradio/tokenizers stack |
| v5 | `HTTP 500` on `/prompt` | Saved workflow JSON (`nodes`/`links`) ≠ API prompt (`{id:{class_type,inputs}}`) |
| v8 | `missing_node_type: MarkdownNote` | UI-only nodes exist in the graph but not in `object_info`; filter them out |
| v9 | `json.load(bytes)` | One-shot typo; the value of printing the response **body** |
| v10 | `KeyError` validating `SaveVideo` | The whole H3 pipeline lives inside `definitions.subgraphs[0]` — skipping an unresolvable node leaves dangling links; you must **flatten** subgraphs, not drop them |
| v11 | (pending) | Flattener + schema-aware widget filter + `extra_data.preview_method` + asset/resolution/RAM alignment |

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

1. Does the v11 flattened prompt pass `/prompt` validation on the first try? (last open 400)
2. How long does a 20-step dense 1344×768 × ~124-frame run take on 2×T4?
3. Does `weight_cache=False` leave enough throughput, or do we need a Dataset-backed boot to
   afford the 18 GiB pinned DiT cache?
4. Optional next: add `H3 MS VAE Split Decode` + `H3 MS Text Encoder Cache`.
