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
| v20 | `SaveAudioAdvanced.execute() missing 'format'` — the kernel sent the *saved-workflow widget value* `{"format":"flac"}` (a dict) where the ComfyUI **API prompt needs the option-key string** `"flac"` for the `IO.DynamicCombo` input; the scheme match fails silently → the input is dropped from the executor's kwargs, and **`node_errors` can't see it** (validation iterates only the expanded schema) | widget forms in a saved workflow ≠ API-prompt input values; DynamicCombo inputs are option-key **strings** with dotted sub-input keys (`format.quality`). A missing-arg crash at execute with clean validation = wrong value *shape*. Since these bugs are invisible to validation, **`local_check.py` (in-repo) runs the executor kwargs path + a real `/prompt` submission locally before every push** |

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

### v17 — image lane + use-case task engine (Phase C start) — RUN ATTEMPTED (partial fail, root-caused)
- **Scope**: the hub stops being video-only. `build_prompt()` branches on `modality`; the image
  branch converts the flat `flux_schnell.json` template (CheckpointLoaderSimple / CLIPTextEncode ×2 /
  EmptySD3LatentImage / KSampler 4 steps cfg 1 euler/simple / SaveImage, 1024²) reusing the same
  `load_info()` schema filter and `widgets_values_named` handling. Preset `image_smoke`;
  `PRUNE_DIRS` gains `checkpoints` so video⇄flux selection prunes the other lane's weights.
- **Smoke schedule** `H3_SMOKE` (default `image,video`): one low-quality smoke per modality,
  image first (fast fail → PNG), wait cell polls a `SMOKE_JOBS` list with per-modality budgets
  (1800 s / 10800 s) and modality-aware assertions (dual-GPU proof only if video ran; the image
  lane legitimately runs single-T4 with Comfy offload).
- **Task engine** `GET /h3api/tasks` + `POST /h3api/task` (8 tasks): `bg_remove`, `extract`
  (rembg u2net mask → `scipy.ndimage` components → element crops + bboxes), `upscale`
  (RealESRGAN_x4plus through Comfy's GPU, synchronous `/history` poll ≤300 s), `video_frames`,
  `video_gif`, `audio_extract`, `audio_trim`, `probe`. Inputs = data URLs or paths confined to
  `/tmp/ComfyUI`; envelope `{ok, outputs, saved, view, meta}` with a 25 MB payload cap.
- **Boot grows to ≈58 GB** (41 GB video + 17.24 GB flux) + 176 MB u2net + 67 MB ESRGAN;
  `/tmp` has 1070 GiB on Kaggle, and write-back drift ⇒ a second dataset upload is expected.
- **Local dry-run: 59/59 ALL TESTS PASSED** before push — including a *real* upscale
  (256² → 1024² CPU), bg_remove alpha extent, extract bbox sanity, path-escape 400, image
  generate through the full handler, and the pub cell printing `TASK SMOKE PASSED`.
- Two dry-run bugs fixed en route (both caught by the harness, not the notebook):
  `T_upscale` posted the node dict **without** the `{"prompt": …}` wrapper (`no_prompt`), and the
  image generate test ran *after* the prune test had legitimately deleted the flux placeholder
  (`value_not_in_list` — Comfy re-scans combo lists, so prune is real) → harness now restores
  placeholders right before that test. Pub GET probes gained retry-once + 30 s timeout +
  `e.reason` (v16's blind `URLError`).

#### v17 run result on Kaggle (31 min, ERROR at the last gate)

**Everything the run was built to prove, proved — except one environment bug:**
- ✅ **Image lane works end to end**: flux downloaded (17.24 GB), `[image smoke] build meta …
  modality: image` → `submitted … node_errors {}` → **PNG out** (`h3_image_00001_.png`).
- ✅ **Video lane still perfect**: `[video smoke]` → `active: 2 ranks` → **206.0 / 205.6 / 204.2 /
  205.0 s/step × 4** → MP4 → `DUAL-GPU CONFIRMED` → `SMOKE PASSED: ['image', 'video']`.
- ✅ **v17's self-test retry fix verified**: pub probes **11/11 green through the tunnel** —
  including `/h3api/outputs` and `/h3api/gpus`, the two that failed in v16.
- ✅ **Two of four task-smoke tasks green live**: `POST task upscale 200` (Real-ESRGAN through
  Comfy's GPU) and `POST task video_frames 200` (3 PNGs).
- ❌ **bg_remove/extract failed**: rembg's install (rc=0) left a **mixed numpy dir** —
  `numpy/_core/strings.py` from a newer numpy imports `_slice`, `umath.py` is 2.1.3-consistent
  and lacks it → `from scipy import ndimage` (and `import rembg`) die with
  `ImportError: cannot import name '_slice' from 'numpy._core.umath'` while plain `import numpy`
  and torch/Comfy work normally. v17's pub cell (correctly) raised on the required image tasks →
  **the sync cell never ran** (no write-back, no manifest probe — cell order matters).

### v18 — numpy probe + auto-heal + TASKS_HEALTHY gate — ✅ VERIFIED ON KAGGLE (first COMPLETE run)
- **Install cell** always prints the pip tail (v17 discarded it on rc=0 — hiding whether the
  resolve moved numpy), then probes the exact imports in a subprocess
  (`from numpy._core.strings import *; from scipy import ndimage; import rembg`); on failure it
  prints numpy diagnostics (`__path__`, `numpy-*.dist-info` list, `_slice` presence per file) and
  **auto-heals** with `pip install --force-reinstall --no-deps numpy==<current version>`, re-probes,
  and finally verifies **in-kernel** → `TASKS_HEALTHY`.
- **Pub task smoke gate**: bg_remove/extract are required only when `TASKS_HEALTHY` (warn-only
  otherwise); upscale stays always-required (it doesn't touch scipy), video tasks stay warn-only →
  **the run can no longer be killed by an environment bug in a secondary feature**, and sync
  always runs.
- `/h3api/tasks` now reports `runtime.healthy` + `runtime.rembg_error` so clients see the state.

#### v18 run result on Kaggle (COMPLETED — first fully green end-to-end run)

**The degradation-first design did exactly its job:**
- ✅ `task deps health: OK` — the subprocess probe passed, **no mixed numpy this boot** (the v17
  directory mix did NOT recur); no heal needed.
- ❌→✅ **in-kernel verify FAILED** but was **root-caused as self-inflicted**: the v18
  `_kernel_verify` *popped* `numpy*` from `sys.modules` and re-imported in the same long-lived
  process, which is itself broken — locally reproduced even on a healthy numpy 2.4.6
  (`ImportError: cannot load module more than once per process`); on Kaggle the re-import walks
  the newer-style `multiarray._override___module__` and dies on
  `AttributeError: 'numpy.ufunc' object has no attribute '__module__'`. So a small install-time
  failure escalated into *total* in-kernel numpy loss (warm-up, api `import rembg`, every task
  re-importing the poisoned numpy) for the whole boot → `TASKS_HEALTHY=False`. **The gate held**:
  upscale (REQUIRED) `200 ok=True`, video_frames `200` (3 outputs), bg_remove/extract
  `WARN tolerated` → **`TASK SMOKE PASSED`** → pub did not raise.
- ✅ **sync finally ran** (the v17 casualty): `cache inventory: 62 files, 59.3 GB` →
  `manifest drift: 5 remote vs 6 local files -> upload` → `staged 6 model files` →
  **`kagglehub upload OK -> adityahalde8777/minimax-h3-model-cache (716 s)`** → dataset
  **version 2** (nvfp4 15.7 G + turbo 4step lora 1.96 G + MANIFEST + placeholder dirs).
- ✅ Everything else green: `DUAL-GPU CONFIRMED`, `SMOKE PASSED: ['image','video']`, pub GET
  self-test all `200`. Run reported **COMPLETED** (kernel done, not ERROR).
- Two client-side flakes exposed: `POST /h3api/settings` had **no retry** (single attempt failed),
  and the bg_remove task POST died with `Network is unreachable` (tunnel blip) — GETs already
  retried, the POSTs didn't.

### v19 — non-destructive in-kernel verify + retry-once POSTs — ✅ VERIFIED ON KAGGLE (Oct 8)
- **`_kernel_verify` no longer touches `sys.modules`**: ONE attempt, print the **first** error +
  traceback tail, return False → `TASKS_HEALTHY=False` → graceful warn-only. The "fix by
  re-importing stem C extensions in the kernel process" idea is gone — the subprocess probe is
  the only place files are re-tested after a heal. This boots the diagnostic that v18 lacked:
  the log will now show *which exact import* fails in-kernel and why.
- **Pub POSTs share one `_post` helper with retry-once** (settings + bg_remove/extract/upscale/
  video_frames task smoke) matching the GET probes' behavior.
- **Phase E happened early**: `build_full.py` moved into the repo (the host restart wiped
  `/tmp/opencode`); reconstructed verbatim from the committed v18 notebook; rebuild verified
  byte-identical against `git show HEAD`.
- Dry-run gate for v19 is the trimmed local check (ast + `_kernel_verify` healthy/failure-path
  micro-test); the full 59/59 `dryrun.py` harness is slated for reconstruction in the repo.

#### v19 run result on Kaggle (COMPLETED — all three success criteria met)

- ✅ **Criterion (a): the precise first error is logged.** Install cell showed `task deps health:
  OK` (subprocess probe on a fresh interpreter) yet **`in-kernel task deps import FAIL:
  ImportError cannot import name '_slice' from 'numpy._core.umath'`** with the traceback tail
  (`numpy/strings/__init__.py` → `numpy/_core/strings.py:22` → `from numpy._core.umath import`)
  → `TASKS_HEALTHY = False`. rembg warm-up + every bg_remove/extract call repeated the same
  import error for the whole boot.
- ✅ **Criterion (b): `TASK SMOKE PASSED` with a clean warn.** upscale `200 ok=True` (REQUIRED),
  video_frames `200` (3 outputs), bg_remove (`Network is unreachable` tunnel blip retried, then
  the numpy import error) and extract FAILED → both `WARN tolerated (install cell reported task
  deps unhealthy after heal)` → no raise. (v18's pop+reimport self-infliction is gone; this time
  the kernel-only failure is a *real* environment condition, exactly what warn-only is for.)
- ✅ **Criterion (c): sync idles.** `cache inventory: 62 files, 59.3 GB` →
  `cache owner: adityahalde8777 (via kagglehub.whoami)` → **`cache dataset up to date (manifest
  match)`** — the v18-created dataset v2 is authoritative, writes now skip. (Read half still
  needs the manual Add Data attach: `/kaggle/input` still held `0 candidate files`.)
- 🎯 **The v17 mystery is solved.** v17 and v19 both showed a *kernel-only* `_slice` error while
  a fresh subprocess imported numpy fine — proportioned exactly by **in-place numpy swap**: a
  subprocess `pip install` (rembg's dep resolve) replaced the numpy wheel files while THIS kernel
  already had numpy 2.1.3 loaded (torch, preflight cell). The disk becomes self-consistent new
  numpy (subprocess probe OK) while the kernel's already-imported `numpy._core.umath` (old, no
  `_slice`) and the new on-disk `strings.py` (imports `_slice`) mix → error. **Reproduced 1:1
  locally** (pip upgraded numpy 2.1.3 → 2.5.3 while loaded → identical in-kernel error; the
  subprocess probe stayed OK), and the **cure verified**: `pip force-reinstall --no-deps
  numpy==<the version the kernel loaded>` (2.1.3) then BOTH probes return OK.

### v20 — Music 3 lane + the proven numpy heal (Oct 8)
- **Install cell heals on EITHER probe**: the subprocess-only heal had a blind spot v19 exposed —
  the disk can be consistent while the kernel is broken. Now the heal fires when the subprocess
  probe OR the in-kernel verify fails, force-reinstalls numpy to the **in-kernel loaded version**
  (`numpy.__version__` in-process, not whatever pip resolved), then re-verifies both. Locally
  verified end-to-end: preload 2.1.3 → rembg resolve upgraded disk to 2.5.3 → subprocess OK /
  kernel FAIL → heal to 2.1.3 → both OK. **v20 verified the heal IN PRODUCTION** (see result).
- **Music lane (MiniMax Music 3)**: `build_prompt()` gains a `music` branch (`_build_music`)
  using **ComfyUI core nodes only** (`comfy_extras/nodes_minimax_music.py` + `nodes_audio.py` +
  `nodes_logic.py`) — no custom pack, no MultiStream, single T4. `audio_minimax_music_3.json`
  (11-node subgraph) is flattened by the **shared `_flatten(wf, info)`**, which was extracted
  from the video flattener verbatim — a local equivalence harness proved the video prompt is
  **byte-identical** after the refactor (27 nodes, identical meta). The music subgraph stores
  inner widgets **positionally** (older graph format, no `widgets_values_named`) → a
  schema-verified `MUSIC_WIDGETS` table promotes them (`""` skips `control_after_generate`).
  Wiring verified node-by-node: instance `switch` → `ComfySwitchNode` selects
  `VAEDecodeAudioTiled` (tile 1536 / overlap 64) vs `VAEDecodeAudio` (lazy — one executes); seed
  → `SeedNode` → `MiniMaxMusic3TextEncode.seed` + `KSampler.seed`; `max_duration` →
  TextEncode + `EmptyMiniMaxMusic3LatentAudio.seconds`; subgraph output link → ComfySwitchNode →
  `SaveAudioAdvanced.audio`.
- **Preset `music_smoke`**: max_duration 5, seed 4242, tiled decode on, `flac`; caption from
  `MUSIC_SMOKE_PROMPT` (lyric-capable). Models: `Comfy-Org/MiniMax-Music-3` (dit fp16 4.91 GB +
  TE pruned int8 convrot 9.20 GB + dav 0.22 GB = 14.33 GB) into the existing
  `diffusion_models/text_encoders/vae` dirs.
- **Smoke schedule default becomes `image,video,music`**: music waits up to 3600 s, then asserts a
  real audio file (`auds` scan — `h3_music_*.flac`). Registry + `/h3api/catalog` (incl.
  `?refresh=1`) carry the music tree; `/h3api/settings` and `/generate` accept `music_smoke`
  (keys `lyrics`/`max_duration`/`tiled_decode`/`format`); docs page documents it.
- **v20 run RESULT (ERROR at the last op — the heal + full music generation PROVEN in
  production)**: install cell printed `task deps health: OK` (subprocess) yet `in-kernel task deps
  import FAIL: … '_slice'` → heal `pip force-reinstall --no-deps numpy==2.1.3` → both probes OK →
  **`TASKS_HEALTHY = True`** (the v20 cure works on Kaggle). All three smokes validated
  (`node_errors: {}`); **music executed for real**: `AR sampling: 57% 72/126 [1.17s/it]`, tiled
  decode ran, lazy `ComfySwitchNode` skipped `VAEDecodeAudio` correctly. But the run ERRORed at
  the last op: `SaveAudioAdvanced.execute() missing 1 required positional argument: 'format'`.
  Root cause (proven locally): `format` is an **`IO.DynamicCombo`**; the API prompt must carry the
  option **key as a plain string** (`"flac"`), and the kernel had sent the widget dict
  `{"format":"flac"}` — the option-key match fails silently → the input is dropped (validation
  iterates only the expanded schema, so `node_errors` stays `{}`) → crash at execute.
- **v21 — the save-node fix, PUSHED**: `_build_music` writes `format` as the plain option-key
  string (mp3/opus also set the dotted `format.quality` sub-input); `_flatten` gained dotted-key
  support. Local CPU ComfyUI proof BEFORE push: flattened prompt has `format:"flac"`, full music
  prompt submits to a real server with **zero node_errors on SaveAudioAdvanced**,
  `build_nested_inputs` nests to exactly `{"format":"flac"}` (mp3 → `{"format":"mp3","quality":"V0"}`),
  and the v20 dict form reproduces the exact drop. Expected: `TASKS_HEALTHY=True`, `SMOKE PASSED:
  ['image','video','music']` with a real `h3_music_*.flac`, bg_remove/extract green.
- **v21 + v21b both died INVISIBLY** (2-byte log `[]`, zero artifacts — not even `state.txt` from
  cell 1): the notebook **never started executing** — the worker died at notebook
  **validation/boot**, not in user code. Root cause (in the builder, not the runtime):
  `build()` emitted cells with **no `id` field**; nbformat's `MissingIDFieldWarning` ("*will become
  a hard error in future nbformat versions*" — that exact warning is visible in the **v20** boot
  log!) became a hard error on the Kaggle runner between v20 and v21. Fix: every cell gets a unique
  id (`h3c00…h3c13`); `nbformat.validate` is now a **zero-warning gate inside `local_check.py`**
  (23/23 — since `21373a9` the gate also rejects a v22.1 regression: install-cell `av` literal
  outside 17..18, or a start cell missing the boot-watchdog markers) and the pushed notebook is
  re-pulled + re-validated before the run is trusted. The
  invisible-death signature (`RUNNING`→`ERROR`, empty log, no artifacts) is now a documented
  diagnostic: zero streams AND zero files — even one a cell writes in its first second — ⇒
  validation/boot death; line-buffering + `state.txt` markers (v21.1) only localize *in-cell* kills.

### v22 — cell-ID fix + transcription task + conditional smokes — PUSHED (results below)
- **Transcription task** (user: "most of the language support"): `POST /h3api/task`
  `{"task":"transcribe"}` via **faster-whisper (CTranslate2, CPU int8, no torch — the GPUs stay
  free for Comfy)** with the full ~100-language Whisper model set (`tiny/base/small/medium/large-v3`,
  default `small`, cached per-run at `models/stt/whisper`). `language: auto|<code>`,
  `task: transcribe|translate`, `format: txt|srt|vtt` (word timestamps + `vad_filter`), data-URL
  plumbing reuses the task engine. Pins `faster-whisper==1.2.1 av==18.1.0` — av must be **17..18**:
  ≥17 satisfies ComfyUI requirements.txt's `av>=17.0.0` (see v22.1), <19 keeps fw 1.2.1's
  `metadata_errors` kwarg. Locally proven: `small` load 3.5 s, 11 s clip
  → 6.5 s, JFK text exact, auto-detect en@0.945, full ~100-language Whisper set; **re-validated
  on av 18.1.0 after the v22 run died at Comfy boot** (JFK flac → 108 chars, en@0.945) — see v22.1.
- **Conditional smokes** ("don't re-test a lane that already ran", user request):
  `H3_SMOKE_MODE=auto|all|none`. Each lane is fingerprinted by its **runtime contract** (resolved
  config + required model files + sizes); a lane whose contract matches the last-passed state is
  **SKIPPED** (`SMOKE SKIPPED (previously proven)`) while still counted in `SMOKE PASSED`. State
  persists at `models/smoke_state.json` and rides the cache-dataset write-back (the sync drift
  decision now includes it) and is read back from the attached dataset + workdir each boot.
  `=all` forces the full suite (use after infra changes), `=none` fast-boots unproven; the
  settings-cell `H3_SMOKE="image,video"` stays the per-modality selector. Shard-tested:
  match→skip, file-size-change→re-verify, mode-all→verify; sync-drift shard: legacy remote
  (no state)→upload, matched→idle, drift→upload.
- **v22 run carries `dataset_sources: [adityahalde8777/minimax-h3-model-cache]`** — the cache
  dataset auto-attaches under `/kaggle/input/minimax-h3-model-cache` on boot (closes 7(b) below).
  This run is the **baseline full-verify** (no stored state yet): all three smokes run and STAMP the
  state, so **v23+ skips proven lanes** automatically.

### v22.1 — v22 run verdict + ComfyUI boot fix — **VERIFIED (ver 24 = FIRST COMPLETE GREEN RUN)**
- **v22 run (ver 23): cell-ID fix WORKED.** 101 KB kernel log (vs the 2-byte `[]` signature),
  `state.txt` milestones through `cell:start:start` — preflight/install/registry/settings/assets/
  convert all executed. The cell-ID root-cause chain is now fully closed.
- **New failure at the start cell:** `RuntimeError: ComfyUI did not listen in time` after a
  **fully silent 363 s window**. Two compounding causes:
  1. Our v22 install cell pinned `av==13.1.0` for faster-whisper, but the ComfyUI main cloned at
     run time declares **`av>=17.0.0`** in its requirements and imports `av` in 6 `comfy_extras`
     modules during boot. pip installed `av-19.0.1` first (ComfyUI reqs), then our pin downgraded
     it in the **same site-packages the ComfyUI server process boots from** → never listened.
  2. The server's stderr went to `/tmp/comfy.log` (a file, lost at kernel end) while the poll loop
     stayed silent → the crash was invisible from the kernel stream.
- **Fix (v22.1):** pin **`av==18.1.0`** — the 17..18 band satisfies both consumers (ComfyUI ≥17,
  fw 1.2.1 <19). Verified locally: this Comfy commit boots on 18.1.0 (and on 19.0.1), and
  faster-whisper transcribes the JFK clip on 18.1.0. The start cell now runs a **boot watchdog**:
  30 s heartbeat printing the last comfy.log line, on failure prints the comfy.log tail + process
  exit code, ONE relaunch, and an extended budget — a Comfy boot failure is never silent again.
- **Verdict (ver 24, 2026-10-08):** **COMPLETE at 92 min** (05:50→07:22 UTC), log 2.26 MB, all 13
  cells executed through `cell:sync:end`. Watchdog heartbeats present; **`ComfyUI local ready`**;
  `TASKS_HEALTHY=True` (t=179 s and task time); **`SMOKE PASSED: ['image','music','video']`** with
  real `h3_music_00001.flac`; **`DUAL-GPU CONFIRMED`** (`active: 2 ranks cuda:0+cuda:1`, 50 blocks,
  28/28 heads); tunnel `https://accurately-initial-grab-quantum.trycloudflare.com` verified public;
  `/h3api/tasks` 200 + `TASK SMOKE PASSED` (bg_remove + extract 200); **sync `kagglehub upload OK`
  (1150 s)** → cache dataset written back incl. `smoke_state.json` (v26 planning page exists for
  STT self-test ver 25 + MOSS-VL `video_qa` lane). Video smoke took ~50 min (slower than v18's
  ~15 min — budget note for future runs). Evidence: `/tmp/opencode/v24out/`.

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
7. ✅→🟡 **Write-back round trip**: **upload half CLOSED in v16** — `kagglehub upload OK ->
   adityahalde8777/minimax-h3-model-cache (530 s)`, 42 GB, dataset created from scratch
   (owner via `kagglehub.whoami`, v15's `SKIP upload` gone). **(a) manifest drift is now
   authoritative: CLOSED — v18 uploaded version 2 on a real `5 remote vs 6 local` drift
   (716 s); v19 probed the populated dataset and idled on `cache dataset up to date (manifest
   match)`. (b) read half still OPEN: **CLOSED by v22** — `dataset_sources:
   [adityahalde8777/minimax-h3-model-cache]` is in the pushed kernel-metadata, so
   `/kaggle/input/minimax-h3-model-cache` auto-attaches on boot (**CONFIRMED ver 24:** cache
   inventory 58 files / 73.8 GB, drift check 6 remote vs 9 local, `kagglehub upload OK` (1150 s)
   → dataset written back incl. `smoke_state.json`).
