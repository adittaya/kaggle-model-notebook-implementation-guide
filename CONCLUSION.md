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

### v15 — enterprise hub API + dataset write-back (Phase B) — ✅ VERIFIED ON KAGGLE
- **Run COMPLETE in 24.9 min, zero stderr errors**: hub came up (`hub API + proxy listening on
  http://127.0.0.1:8190`), tunnel targeted `:8190`, `Verified public endpoint:
  https://costa-ltd-kelly-hosting.trycloudflare.com`, full READY block with `/h3api/docs`,
  `/h3api/health`, `/h3api/generate` URLs printed.
- Smoke path identical to v14 through the new `convert` cell: subgraph expanded → 21 nodes,
  3 UI-only notes skipped, flattened 24 nodes, turbo armed, `prompt_id 59047396…`,
  `node_errors {}`; **207.2–208.2 s/step × 4**; `DUAL-GPU CONFIRMED: active: 2 ranks`;
  MP4 `MiniMax_H3_00001_.mp4` out.
- **One bug found by the run itself**: sync printed
  `SKIP upload: cannot determine dataset owner - continuing` — modern Kaggle notebooks
  authenticate via a **token file** (`KAGGLE_API_V1_TOKEN`), *not* `KAGGLE_USERNAME`/
  `kaggle.json`, so the owner check found nothing. The never-fail design worked (run
  completed), but write-back silently skipped → **v16 fix**: owner via `kagglehub.whoami`
  (native notebook auth) + env/kaggle.json/embedded-owner fallbacks, upload via
  `kagglehub.dataset_upload` (classic CLI second), auth diagnostics printed, plus a **live
  endpoint self-test** in the pub cell (health/catalog/settings/jobs/outputs/gpus/docs,
  proxied `/object_info`, 404 envelope, POST settings — all through the tunnel).
- v15 local dry-run before push: **all tests green** — every endpoint, WS `101`, chunked POST,
  import `201`, traversal rejected `400`, prune verified, `first_frame` → 24-node prompt
  validated, sync `dry` staged 6 model files without uploading.

### v16 — write-back fixed, first cache dataset created — ✅ VERIFIED ON KAGGLE
- **Run COMPLETE in 32.9 min (1972 s), zero errors** — includes the 530 s first-time upload.
- **The v15 bug is fixed and proven live**: `auth env: KAGGLE_USERNAME=False KAGGLE_KEY=False
  KAGGLE_API_V1_TOKEN=True kaggle.json=False` → `in-kaggle-notebook: True` →
  `cache owner: adityahalde8777 (via kagglehub.whoami)` — exactly the token-file auth story
  v15 taught us, with `kagglehub.whoami` doing the work.
- **First write-back succeeded**: `cache inventory: 55 files, 42.0 GB` → `staged 5 model files`
  → `Uploading Dataset …` → **`kagglehub upload OK -> adityahalde8777/minimax-h3-model-cache
  (530 s)`** (~80 MB/s; largest files 21.0 + 15.7 + 2.8 + 2.0 + 0.6 GB). Dataset **created**
  from scratch by `dataset_upload` — no manual `kaggle datasets create` needed.
- **Probe behaves correctly**: `probe: kagglehub probe: BackendError; cli rc=1 403` →
  `remote MANIFEST unavailable -> upload (first run or probe failed)` — correct, the dataset
  didn't exist at probe time. The `403` on the classic CLI inside a notebook is the known
  token-file-auth limitation (kagglehub path works; CLI is fallback only).
  *Open: confirm the second boot probes `MANIFEST.json` OK and skips the upload.*
- **Pub self-test through the tunnel**: `200` on health/catalog/settings/jobs/docs,
  `200` on proxied `/object_info`, `404 /h3api/nope` with the JSON envelope,
  `200 POST /h3api/settings`. **7/9 + POST green.**
- **Two flaky lines**: `URLError /h3api/outputs` and `URLError /h3api/gpus` — the only two
  endpoints that do real OS work (output-dir scan, `nvidia-smi` subprocess), and the only two
  requests that hit the 20 s client timeout, with neighbors before and after passing ⇒ most
  likely a transient trycloudflare stall; but the handler printed only `URLError` (reason
  swallowed) so it's undiagnosed → **v17 fix: retry-once, 30 s timeout, print `e.reason`**.
- Smoke path unchanged and healthy: `prompt_id 1964b47f…`, `node_errors {}`,
  **194.4–195.9 s/step × 4** (fastest measured: v14 210 → v15 207 → v16 194–196),
  `DUAL-GPU CONFIRMED: active: 2 ranks`, `smoke status: success`, MP4 out, READY +
  `Verified public endpoint: https://claimed-maintain-kate-stroke.trycloudflare.com`.

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
7. ⬜→✅ **Write-back round trip**: **upload half CLOSED in v16** — `kagglehub upload OK ->
   adityahalde8777/minimax-h3-model-cache (530 s)`, 42 GB, dataset created from scratch
   (owner via `kagglehub.whoami`, v15's `SKIP upload` gone). Still open: (a) second-boot probe
   must find `MANIFEST.json` and **skip** the re-upload (v16's `BackendError/403` probe was
   correct first-run behavior — verify in v17), (b) the read half: attach the dataset to the
   kernel once via **Add Data** so `/kaggle/input` populates and later boots symlink models
   instead of downloading from HF.
