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
| (v15) | *build-time, caught locally:* `b"\r\n"` inside the build script's triple-quoted cell strings was **eaten as real control chars** by the outer string → the generated cell had an unterminated bytes literal | escape sequences in generated notebook code must be **doubled** in the build script (`\\n`); `ast.parse` **every generated cell**, not just the build script — and note Kaggle CLI's `datasets version` takes no `-d` (reads `dataset-metadata.json` from `-p`) |

### Hard numbers measured on the Kaggle image
- 2 × Tesla T4 (14.6 GB each), topology `PHB`, **host ≈ 10 GB/s vs p2p ≈ 9.3 GB/s**
  ⇒ `exchange="host"`, `exchange_chunks=8`.
- **System RAM = 31.3 GB** ⇒ all pinned weight caches OFF (node-pack README: 32 GB class ⇒
  caches disabled; the DiT cache alone wants 18 GiB).
- Resolution table from the template: **0.98 MP = 1344×768** (official 768p) — the target quality row.

### v14 fast smoke — VERIFIED on Kaggle (Phase A complete)
- **Whole run 25.6 min** (first log t=6 s, last t=1538 s), zero stderr errors/warnings.
- Assets: **5/5 `OK <bytes>`** = 41.03 GB (20.97 DiT + 15.69 TE + 2.81 video VAE + 0.61 audio VAE
  + 1.96 turbo LoRA), downloaded in minutes with the HF token (no dataset attached yet);
  registry auto-listed 39 core files + 100 community repos.
- Turbo path armed: `PrimitiveBoolean ['1021'] → steps 4 + LoRA`; `/prompt` accepted
  (`prompt_id 67580883…`, `node_errors {}`).
- **Sampling: 209.6–210.9 s/step × 4 steps** — vs 861 s/step dense @0.98 MP ⇒ **4.1× faster per
  step** (turbo 5× fewer steps + 0.4 MP is 2.45× fewer pixels). Full fast smoke ≈ 14 min sampling.
- Exchange cost measured: `exchange host ×8` ≈ **8.5 s + 3.9 s of each ~210 s step (≈6 %)**,
  25.74 GiB moved/step ⇒ the slow dense step is **compute-bound on T4s**, not split overhead.
- Dual-GPU again: `2 rank(s): cuda:0 (primary), cuda:1` → `active: 2 ranks … 50 blocks, heads
  28/28`; during sampling GPU0 `100 % / 7657 MiB / 64.9 W`, GPU1 `100 % / 4645 MiB / 62.1 W`.
- Output produced: `/tmp/ComfyUI/output/video/MiniMax_H3_00001_.mp4`; READY block printed the
  public Base URL (quick-tunnel URLs are ephemeral per session).

### v15 — enterprise hub API + dataset write-back (Phase B, pushed)
- **One Base URL serves three planes**: ComfyUI web UI, native `/prompt` API, and the
  `/h3api/*` management hub (docs/health/catalog/models/settings/select/download/generate/
  jobs/outputs/free/gpus/import) — all behind one cloudflared tunnel to a stdlib proxy on
  `:8190` with raw WebSocket passthrough.
- Settings surface exposed end-to-end (preset/set/seed/frame conditioning); model switching
  **downloads the new set and prunes superseded files** so disk never bloats.
- Kaggle Dataset write-back: hardlink staging (≈0 extra disk), manifest-drift detection,
  **200 GB cap → auto-skip**, every failure path prints and continues (never fails the run).
- Local dry-run: **all tests green** — every endpoint, WS `101`, chunked POST, import `201`,
  traversal rejected `400`, prune verified, `first_frame` → 24-node prompt validated,
  sync `dry` policy staged 6 model files without uploading.

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
   ✅ v14: same validation reproduced **locally** (CPU ComfyUI dry-run: `prompt_id 918c50eb…`,
   `node_errors: {}`, turbo path armed) before pushing.
1.5 ✅ **Fast path timing (v14 measured)**: **209.6–210.9 s/step × 4 steps**, whole run
   **25.6 min** end to end (init ~14 min included) — well under the 3 h cap. MP4 produced.
2. ✅ **20-step dense 1344×768 on 2×T4 ≈ 861 s/step → ≈4.8 h sampling** (v12, first measurement).
   Why ~17× slower than the pack's 2×5060 Ti reference (42–49 s/step)? **v14 answered the big
   half:** exchange accounts for only ≈6 % of a step (8.5+3.9 s of ~210 s), so the dense slowness
   is **T4 compute**, not split overhead — and the turbo+480p path gives a measured 4.1× per-step
   speedup (861 → 210 s/step).
3. ✅ **Is the split actually on both T4s?** — **YES, verified live in v13 and again in v14**:
   `[GPUs] dit: 2 rank(s): cuda:0 (primary), cuda:1` + `active: 2 ranks cuda:0+cuda:1,
   50 blocks, heads 28/28` + both cards at `100 % util / ~62–65 W`, 7.7/4.6 GiB resident;
   no `UNSPLIT`. The final assertion confirms it at smoke end.
4. Does `weight_cache=False` leave enough throughput, or do we need a Dataset-backed boot to
   afford the 18 GiB pinned DiT cache? (v14 ran fine without the cache; write-back now exists
   so a future test can afford it.)
5. ✅ Optional speed lever: the template's `turbo_mode` switch → 4-step turbo lora path —
   **armed and verified in v14** (`PrimitiveBoolean ['1021']`, 210 s/step). Remaining levers
   (untested): `H3 MS VAE Split Decode`, `H3 MS Text Encoder Cache`.
6. ⬜ **First/last-frame conditioning at runtime**: the prompt structure validates locally
   (LoadImage → subgraph input, `node_errors {}`), but no Kaggle run has yet *executed* an
   image-conditioned generation end to end.
7. ⬜ **Write-back round trip**: v15 creates/refreshes `MiniMax H3 model cache` — the loop
   closes when a later boot actually symlinks models out of `/kaggle/input` (needs the dataset
   attached once via Add Data).
