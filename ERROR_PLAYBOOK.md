# Error Playbook — Failure Chains We Hit and Their Fixes

Three chains are documented: **(A) model downloads** (WanGP lane, retired) and
**(B) ComfyUI `/prompt` submissions** (current lane, v3 → v11), plus **(C) task-engine /
local dry-run traps** (v17). All are reusable patterns.

## (A) Kaggle / Hugging Face downloads

| Run | Symptom | Root cause | Fix applied |
|---|---|---|---|
| v2 | `RuntimeError: pinned FL2VA definition not discovered` | Filter matched on model *name* containing "Kaggle" — it didn't | Match by `model_type` prefix (`h3_kaggle_`) |
| v3 | `TypeError: sequence item 0: expected str instance, GenerationError found` | `result.errors` were objects, never surfaced | Render `e.message` + `e.stage` |
| v4 | `.../_download_xxx.incomplete` no such file | WanGP's patched `http_get` masked the real error | Predownload with `hf_hub_download(local_dir=...)` |
| v5 | `cannot find the requested files in the local cache` | HF `hf_xet` bridge unreachable from Kaggle egress | `HF_HUB_DISABLE_XET=1` + `HF_HUB_DISABLE_HF_TRANSFER=1` |
| v6 | **✅ assets fetched** | — | stable stage |

**Rules:** first fetch = `hf_hub_download` with xet disabled, out-of-band, into the exact layout
the runtime expects. Never `pip install -U huggingface_hub` while doing this (see B/v3).

## (B) ComfyUI `/prompt` — the v3 → v11 chain (current lane)

| Run | Symptom | Root cause | Fix |
|---|---|---|---|
| v3 | `RuntimeError: ComfyUI did not listen in time` | notebook ran `pip install -U huggingface_hub` → 2.1.x → broke ComfyUI's pinned gradio/tokenizers/diffusers | remove the upgrade line entirely |
| v5 | `HTTP 500` on `POST /prompt` | posted the **saved workflow** (`nodes`/`links`/`groups`) | convert to **API prompt** `{id:{class_type,inputs}}` |
| v6 | `HTTP 400`, no body shown | debug code had its own bug (`e` vs `ee`) | print `ee.read()` in the handler |
| v7 | `NameError: name 'e' is not defined` | same handler bug masked the answer | fix the variable name |
| v8 | `missing_node_type: MarkdownNote` (node `#116`) | UI-only annotation nodes aren't in `object_info` | fetch `GET /object_info`, skip unregistered types |
| v9 | `AttributeError: 'bytes' object has no attribute 'read'` | `json.load(urlopen(...).read())` | use `json.loads(...)` |
| v10 | `prompt_outputs_failed_validation` → `KeyError prompt[o_id]['class_type']` validating `SaveVideo` | skipped the UUID node → dangling link. The UUID is a **subgraph** holding the entire 21-node H3 pipeline | **flatten subgraphs**, never drop them |
| v11 | 6 × `value_not_in_list` (`unet_name ... not in []`, `vae_name ... not in ['pixel_space']`, clip/lora lists empty) + `required_input_missing: values.a` | (a) `python -m huggingface_hub.cli.download` has **no `__main__` guard** → module imported, exit 0, **zero files downloaded**, all model folders empty. (b) `values.a` autogrow slot exists only after schema finalization, so the schema filter dropped a required link | in-process `snapshot_download` + per-file size assert; include every **linked** input regardless of static schema |
| v12 | healthy run at **861 s/step** (tqdm ETA ≈4.8 h / 20 steps) vs pack reference 42–49 s/step on 2×5060 Ti (~17× off); its **60-min wait cap would false-fail** the job | perf anomaly to diagnose (T4 compute vs exchange vs off-plan); timeout budgets must come from measured throughput, not estimates | surface `[MultiStream] step:` lines live; wait cap sized from measured s/it (v13: 6.5 h); fail *fast* on `UNSPLIT` |
| v22 | `RuntimeError: ComfyUI did not listen in time` — **101 KB log** (cell-ID fix WORKED: cells ran through assets/convert), `state.txt` ended at `cell:start:start`, then a **fully silent 363 s window** between `Comfy pid 252` and the raise | install cell pinned `av==13.1.0` (for faster-whisper 1.2.1) but ComfyUI main's `requirements.txt` now requires **`av>=17.0.0`** and imports `av` in 6 `comfy_extras` modules at boot. pip installed `av-19.0.1` during the ComfyUI reqs step, then our pin **downgraded it to 13.1.0 in the same site-packages the ComfyUI server process boots from** → server never listened. The crash itself was invisible: the server's stderr went to `/tmp/comfy.log` (a file, unreachable post-run) while the kernel's poll loop was silent for the whole budget | pin **`av==18.1.0`** (band 17..18 — ≥17 satisfies ComfyUI, <19 keeps fw 1.2.1's `metadata_errors` kwarg; both verified locally: this Comfy commit boots on 18.1.0, JFK flac transcribes on 18.1.0). Start cell now runs a **boot watchdog**: 30 s heartbeat printing the last comfy.log line, on failure prints the **comfy.log tail + process exit code**, ONE relaunch for transient crashes, and an extended wait for slow boots — a Comfy boot failure can never be silent again |

> Sanity signal you can use too: v11 finished in **111 s** — impossible for a multi-GB download,
> and the download command printed *nothing*. A subprocess with no output and exit 0 is a lie.

### Supporting bugs found by inspection before they fired

| Risk | Symptom it would have caused | Fix in v11 |
|---|---|---|
| Wrong VAE filename (`fp16` vs `int8_convrot`) | `400 value not in list` on `VAELoader` | download exactly the widget's filename (+ the turbo lora) |
| Template default `megapixels=0.4` | silent 864×480 instead of ~1K | force `0.98` → 1344×768 |
| `weight_cache=True` on 31.3 GB RAM | OOM / pinned-RAM refusal mid-run | `weight_cache=False` |
| Missing `extra_data.preview_method` | preview crashes under DynamicVRAM (per node-pack README) | send `latent2rgb` in the POST |
| Non-string link ids in the prompt | `KeyError prompt[o_id]` (JSON keys are always `str`) | `["1017", 0]` everywhere |
| Frontend-only widget keys (`fixed`, `control_after_generate`) | `invalid_input_type` / unknown input | schema-filter every widget name |
| Broken cloudflared fetch (curl + stale pinned URL) | tunnel never starts, no READY block | `latest/download/...` with fallback + size check |
| Generated cell had `SyntaxError: unterminated string literal` at `b"\r\n"` | the build script's **triple-quoted** string consumed `\r\n`/`\n` as *real* control chars, so the emitted notebook code contained raw newlines inside a bytes literal — the build script itself still parsed fine | double every escape meant for the notebook (`b"\\r\\n"` in the build script); **`ast.parse` every generated cell**, not just the build script |
| `kaggle datasets version -d <ref>` → CLI error / wrong target | `-d` on `datasets version` means `--delete-old-versions`, not the dataset — the command reads `dataset-metadata.json` from the `-p` folder | write `{title, id, licenses}` into `dataset-metadata.json` in the staging folder; `list` CSV needs `-v` (not `--csv`); existence probe = `kaggle datasets metadata <owner>/<slug>` |
| v15 sync cell: `SKIP upload: cannot determine dataset owner - continuing` (run healthy, write-back silently skipped) | **modern Kaggle notebooks authenticate via a token file** (`KAGGLE_API_V1_TOKEN`), *not* `KAGGLE_USERNAME`/`KAGGLE_KEY`/`~/.kaggle/kaggle.json` — the classic owner check finds nothing; the classic `kaggle` CLI also gets `403` in-kernel | owner = `kagglehub.whoami()` (native in-notebook auth) → env pair → kaggle.json (multi-path incl. `KAGGLE_CONFIG_DIR`) → **embedded kernel owner** (public, build-time from kernel-metadata `id`); upload via `kagglehub.dataset_upload` (creates/versions natively), classic CLI as second fallback; print auth **diagnostics** (booleans only, never secrets) every run so the next log shows which sources exist — **v16 verified live**: `via kagglehub.whoami` + `upload OK (530 s)` |
| v16 self-test: `URLError /h3api/outputs` + `URLError /h3api/gpus` (the only 2 of 9 GETs to fail; neighbors 200) | those two handlers do real OS work (output-dir `rglob`, `nvidia-smi` subprocess) and sat behind a transient **trycloudflare** stall → the 20 s client timeout fired; the print showed only `URLError` because the exception's `reason` was never surfaced | v17: **retry once** on `URLError`, raise the GET timeout to 30 s, and print `e.reason` so a slow handler and a tunnel drop are distinguishable; re-verify next run |
| v16 sync probe: `kagglehub probe: BackendError; cli rc=1 403` → `remote MANIFEST unavailable -> upload` | **expected first-run behavior**: the dataset didn't exist when the probe ran (`403` = classic CLI's known in-kernel token gap; `BackendError` = dataset not yet created) — the upload then created it | no fix needed; **verify in v17** that the second boot probes `MANIFEST.json` OK and prints `cache dataset up to date (manifest match)` (i.e. no 42 GB re-upload) |

## (C) Task engine + local dry-run traps (v17)

| Where | Symptom | Root cause | Fix |
|---|---|---|---|
| **v17 Kaggle run (31 min, ERROR at the last gate)** | every lane green (image PNG, video MP4 204–206 s/step × 4 dual-GPU, `SMOKE PASSED`, pub probes **11/11 `200`**, tasks upscale/video_frames `200`) — then pub task smoke raised `task smoke failures: ['bg_remove', 'extract']`, **skipping the sync cell**: `ImportError: cannot import name '_slice' from 'numpy._core.umath'` on `from scipy import ndimage` | **mixed numpy dir on the Kaggle image**: `numpy/_core/strings.py` is from a *newer* numpy (its umath import list includes `_slice`) while `umath.py` is the 2.1.3 file (verified: v2.1.3 sources contain no `_slice` anywhere) → plain `import numpy`, torch and Comfy work, but anything loading `numpy._core.strings` (scipy, rembg) dies. The rembg pip output was **discarded on rc=0**, so whether the resolve touched numpy was invisible | v18: install cell **always prints the pip tail**, probes the exact imports in a subprocess (`from numpy._core.strings import *; from scipy import ndimage; import rembg`), on failure prints numpy diagnostics (`__path__`, `numpy-*.dist-info` list, `_slice` per file) and **auto-heals** with `pip force-reinstall --no-deps numpy==<ver>` (a same-version resync rewrites every file from one wheel, re-consistent by construction), re-probes, then verifies **in-kernel** (drop stale `numpy*` from `sys.modules` first) → `TASKS_HEALTHY`; pub requires bg_remove/extract **only when healthy**, so an env bug can never again skip write-back |
| local box | `import rembg` → default model download of **`bria-rmbg` (1.02 GB)** → process `SIGKILL`'d on the 8 GB dev machine | rembg's own default model is a big commercial one; the local box cannot hold it | pin **`rembg[cpu]==2.0.85`**, always construct sessions with an **explicit small model** (`u2net`, 176 MB) and `REMBG_HOME` inside `models/` — never accept library defaults for weights |
| dry-run | `POST /h3api/task upscale` → `{"error": "comfy rejected upscale prompt", "detail": "No prompt provided"}` | the handler built the node dict correctly but posted it as the **whole body** instead of `{"prompt": <node dict>}` — Comfy's API takes a wrapper | wrap every manual `/prompt` post as `{"prompt": ...}`; the 503-vs-502 body printing made the cause obvious in one look |
| dry-run | `POST /h3api/generate preset=image_smoke` → `400 value_not_in_list: ckpt_name 'flux1-schnell-fp8.safetensors' is not in [...]` | the harness's own **select-prune test ran first** and legitimately deleted the flux placeholder (prune keeps only the active modality's files; Comfy re-scans combo lists, so the file really vanished) | restore the placeholder immediately before the image test — and remember: *test order matters* when your own feature is pruning files mid-run; on the real kernel this is correct behavior (flux must not linger in video mode) |
| build script | `H_catalog` refresh branch would have raised `NameError: reg` | the branch used `reg.get("image_repo", ...)` **before** `reg = json.load(...)` — a variable that only exists after the load | load the registry first (`reg0`) inside the refresh branch; ordering bugs in error paths only fire when `?refresh=1` is called — which is exactly when users call it |
| local box | first rembg warm-up downloads weights through a **symlinked** `models/` dir | `/tmp/ComfyUI → /tmp/opencode/ComfyUI` symlink; relative/`os.path` writes resolve through it fine, but inventory code that string-compares paths may not | keep `REMBG_HOME` under the resolved `models/` path and check `Path.resolve()` when comparing |
| **v18 Kaggle run (COMPLETED, but bg_remove/extract down all boot)** | `task deps health: OK` (subprocess probe passed — **no mixed numpy this boot**) yet the **in-kernel** verify failed with `AttributeError: 'numpy.ufunc' object has no attribute '__module__'` → `TASKS_HEALTHY=False` → warm-up, api `import rembg` and every task died for the whole boot | **self-inflicted `_kernel_verify` bug**: it *popped* `numpy*` from `sys.modules` and re-imported in the same long-lived process. Pop-and-reimport of a stem C-extension module is broken by itself: locally reproduced on a healthy numpy 2.4.6 (`ImportError: cannot load module more than once per process`); on Kaggle the re-import walks the newer-style `multiarray._override___module__` and fails setting `ufunc.__module__` on the re-init'd objects → the *verified-healthy-subprocess* state and the *broken-kernel* state diverge | v19: **never touch `sys.modules` in `_kernel_verify`** — ONE attempt, print the FIRST error + traceback tail, return False (graceful `TASKS_HEALTHY=False`). The subprocess probe is the only place on-disk files are re-validated after a heal; a kernel process only *reports*, it does not reload C extensions. The `TASKS_HEALTHY` gate already made this non-fatal (upscale REQUIRED green, sync ran, dataset version 2) |
| **v19 Kaggle run (COMPLETED — all three diagnostic goals met)** | `task deps health: OK` (subprocess probe on a fresh interpreter) yet **`in-kernel task deps import FAIL: ImportError cannot import name '_slice' from 'numpy._core.umath'`** (first error + traceback tail printed — exactly the diagnostic v18 lacked) → `TASKS_HEALTHY=False` → bg_remove/extract FAILED (after retry-once) but `WARN tolerated` → `TASK SMOKE PASSED`; **sync idled: `cache dataset up to date (manifest match)`** | **the mixed numpy is an IN-PLACE SWAP, not a bad image**: a subprocess `pip install` (rembg's dep resolve) replaces the numpy wheel files while THIS kernel already pre-loaded numpy 2.1.3 (torch preflight). Disk then imports self-consistently (new numpy → subprocess probe OK) while the kernel's already-imported `umath` (old, no `_slice`) + new on-disk `strings.py` (imports `_slice`) mix → kernel-only failure. Reproduced 1:1 locally (in-place upgrade 2.1.3→2.5.3 while loaded → same error; subprocess stayed OK) | **v20: heal fires when EITHER probe fails**, force-reinstalling numpy to the version the KERNEL loaded (`numpy.__version__` in-process), then re-verifies both — the cure itself verified locally (after heal: subprocess OK + in-kernel OK, `TASKS_HEALTHY=True`). Lesson: a subprocess-only probe validates disk files, not the long-lived process's loaded modules — probe both |
| **v20 Kaggle run (ERROR at the last op — numpy heal PROVEN in production)** | install: `task deps health: OK` (subprocess) yet `in-kernel task deps import FAIL: … '_slice'` → **heal `force-reinstall --no-deps numpy==2.1.3` → both probes OK → `TASKS_HEALTHY = True`**; image+video smokes green; **music smoke EXECUTED** (`AR sampling: 57% 72/126 [1.17s/it]`, tiled decode, lazy `ComfySwitchNode` correctly skipped `VAEDecodeAudio`) but the run died at the last op: `SaveAudioAdvanced.execute() missing 1 required positional argument: 'format'` | SaveAudioAdvanced.`format` is an **`IO.DynamicCombo`**: the ComfyUI API prompt must carry the option **KEY as a plain string** (`"flac"`); the kernel was sending the UI-widget dict `{"format":"flac"}`. `get_finalized_class_inputs` uses that value to *match the option* — a dict never matches → the input is **silently dropped** (validation iterates only the expanded schema, so `node_errors` stays empty) → `execute()` gets kwargs without `format` → the crash. Proved on a local CPU ComfyUI: dict form drops; string form nests (`build_nested_inputs`) to exactly what `execute()` reads | **v21: `_build_music` writes the option-key string** (`"flac"`; mp3/opus also set the dotted `format.quality` sub-input) and `_flatten` gained dotted-key support. Local proof before push: flattened prompt has `format:"flac"`, full prompt validates on a real server with **zero node_errors on SaveAudioAdvanced**, and the v20 dict form reproduces the exact drop. Lesson: widget-value shapes (saved-workflow dicts) ≠ API-prompt input values — DynamicCombo inputs are option-key strings (+ dotted sub-input keys); a missing-input that only fires at execute means the value form is wrong |
| **v21 + v21b Kaggle runs (ERROR with NO log at all — 2-byte `[]`)** | both burned ~50 min at `RUNNING` then ERRORed with a **2-byte log** (`[]`): zero stdout/stderr stream events AND zero `outputs/` artifacts — not even `state.txt`, which the *secrets* cell (cell 1) writes before anything else inside the notebook can fail | a kernel that produces zero streams and zero artifacts — not even a file a cell would create in its first second — **never started executing cells**: the worker died at notebook validation/boot, not in user code. A SIGKILL mid-run can eat buffered stdout (v21's line-buffering hardening was aimed at that) but cannot erase a file already written to `/kaggle/working` | root cause in the builder: `build()` emitted cells with **no `id` field** — and nbformat warns *`MissingIDFieldWarning`… "will become a hard error in future nbformat versions"* (that exact warning is visible in the **v20 boot log**!). v20 ran while the runner tolerated it; the runner enforced it by v21 → notebook refused to start → empty log, no artifacts. **v21c/v22: every cell gets a unique `id` (`h3c00…h3c13`)**; `nbformat.validate` now passes with **zero warnings**, and the pushed notebook is re-pulled + re-validated before the run is trusted |

## Debugging signals that actually solved things

- **Always print the HTTP error body.** v5–v7 were blind; every subsequent fix came from the body.
- **A RUNNING→ERROR kernel with a 2-byte log (`[]`) and zero output artifacts** ⇒ the notebook
  never started executing: no streams (not even pip/torch prints), no files (not even one a cell
  would write in its first second). Debug the **notebook file + runner** — cell IDs
  (`MissingIDFieldWarning` becomes a hard error; v21/v21b died exactly this way), notebook
  metadata, validation — not user code. Line-buffering + `state.txt` markers (which this repo has)
  only localize *in-cell* deaths; their *absence* is the fingerprint of a validation/boot death.
- `missing_node_type` ⇒ node is frontend-only or the custom pack failed to load → diff against
  `GET /object_info`.
- `KeyError prompt[o_id]` in `validate_inputs` ⇒ dangling link: unflattened subgraph, skipped
  node, or int-vs-string id mismatch.
- `value_not_in_list ... not in ['pixel_space']` ⇒ `models/vae` is **empty** (`pixel_space` is a
  built-in pseudo-VAE, `nodes.py` appends it to every VAE list), so your download never landed.
- **A node dies at EXECUTE with "missing required positional argument" while `node_errors` was `{}`**
  ⇒ wrong value SHAPE for a DynamicCombo / dynamic input, not a missing key: the executor
  (`get_finalized_class_inputs`) matches the input VALUE against the option list to decide which
  sub-inputs to inject, so a mismatched value silently drops the input after validation. API
  prompts must carry the **option-key string** (`"flac"`), plus dotted sub-input keys
  (`format.quality`); saved-workflow widget dicts (`{"format":"flac"}`) are UI form, not API form.
  `local_check.py` (in-repo: `--live <comfy-server>` `--comfy <ComfyUI-dir>`) exercises this exact
  executor kwargs path + real `/prompt` before every push, so this bug class dies locally, not on a
  45-min Kaggle run.
- A whole notebook that finishes in ~110 s, or a download subprocess with **no output and
  exit 0** ⇒ it did nothing. Verify files and sizes, never trust the exit code.
- **Config ≠ execution for multi-GPU**: `device_count: 2` and `second_gpu=-1` only show intent.
  Runtime proof is `[GPUs] dit: … cuda:0, cuda:1` + `[MultiStream] active: 2 ranks` in
  comfy.log; `running UNSPLIT` / `running unsplit on 1 GPU` means the split silently fell back
  to one card. The notebook's wait cell now asserts this (raises unless `active: 2 ranks` is
  present) and tails `nvidia-smi -l 10` samples as `[gpu]` lines so utilization per card lands
  in the Kaggle output. **Verified working in v13:** `2 rank(s): cuda:0 (primary), cuda:1` +
  `active: 2 ranks` + both cards `100 % / 65.9 W` (T4's ~70 W ceiling) with 10.4/8.6 GiB
  resident — the canonical healthy pattern to compare against.
- **Fast-smoke healthy reference (v14, 25.6-min run):** 5/5 `OK <bytes>` (41.03 GB) ->
  `turbo path armed: PrimitiveBoolean ['1021']` -> `submitted: ... node_errors {}` ->
  `active: 2 ranks` -> ~210 s/step x 4 -> `smoke status: success completed: True` ->
  `found: /tmp/ComfyUI/output/video/MiniMax_H3_00001_.mp4` -> READY block with Base URL.
  If your run deviates from that sequence, start from the first divergence.
- **Write-back healthy tail (v16 — PROVEN on Kaggle):** `auth env: KAGGLE_USERNAME=... kaggle.json=...`
  (booleans) → `in-kaggle-notebook: True` → `cache owner: <user> (via kagglehub.whoami)` →
  `cache dataset: <owner>/minimax-h3-model-cache | probe: ...` → `staged N model files` →
  `kagglehub upload OK -> <ref> (<seconds> s)`. Any `SKIP upload:` line means the owner chain
  failed — see the v15 token-file row above. v16 measured: 5 files / 42 GB staged, upload
  **530 s** at ~80 MB/s, dataset created from scratch. On the *second* boot the probe should
  find the remote `MANIFEST.json` and print `cache dataset up to date (manifest match)` instead
  of uploading again (v17 checkpoint).
- **v17/v18 multi-modality healthy sequence (PROVEN on Kaggle):** registry prints image/extras
  sections → assets print `OK <bytes>` for 5 video files **+ flux + u2net + RealESRGAN** →
  `[image smoke] build meta {… "modality": "image" …}` → `submitted … node_errors {}` →
  **PNG found** → `[video smoke] submitted` → `active: 2 ranks` → ~195–206 s/step × 4 → MP4 →
  `SMOKE PASSED` → tunnel → self-test probes all `200` (retry-once, `e.reason` printed on
  retry) → **`TASK SMOKE PASSED`** → sync (drift → upload, or `manifest match` on a populated
  dataset) → READY. **v17 matched up to the last step** (bg_remove/extract — chain (C) above);
  **v18 matched fully** (with bg_remove/extract warn-only against the self-inflicted numpy bug,
  sync uploaded version 2 in 716 s); **v19 = the same sequence with a clean warn** (bg_remove/
  extract FAILED on the in-place-numpy-swap but `WARN tolerated`), and **sync idled on `manifest
  match`** (no drift); **v20 adds the third modality to that sequence** (music smoke → audio
  file assertion) and **delivered `TASKS_HEALTHY=True` via the either-probe heal (PROVEN in
  production)** but ERRORed at the music save step (the v20 chain above) — **v21 re-runs the full
  sequence with the save-node fix**. When the log
  matches the healthy sequence up to one line, that line *is* the bug — a missing flux line
  means the image schedule didn't load (`H3_SMOKE`); a missing PNG after image smoke means the
  checkpoint never landed (`value_not_in_list` on `ckpt_name`); a missing FLAC after music smoke
  means the MiniMax-Music-3 assets never landed or the module is wrong; a missing `format` on
  `SaveAudioAdvanced` at the **execute** stage (node_errors still `{}`) means a DynamicCombo
  input was sent in the wrong value form (v21 lesson — send the option-key string, not a dict);
  a `WARN tolerated` row
  means the environment (not the feature) is broken — stay in warn-only and let the run finish.
- **Dry-run the notebook cells locally before pushing**: run every cell against a local ComfyUI
  (CPU build) — registry/settings/assets/convert/submit/API all execute for free and `/prompt`
  validation is the real `validate_prompt`. The v15 harness goes further: after the cells it
  **exercises every `/h3api/*` endpoint, the HTTP proxy, a WebSocket `101` handshake, a chunked
  POST, and the sync cell's `dry` policy** — all green before the push. Gotcha: on a CPU-only
  torch box Comfy crashes at import with
  `Torch not compiled with CUDA enabled` unless you start it with `--cpu` (the harness patches
  the start cell; the Kaggle cell is untouched because T4s exist there).
- **Size timeouts from measurements, not estimates**: v12 ran at 861 s/step (≈4.8 h job) under a
  60-min wait cap — the cap would have killed a healthy run. A tqdm line like
  `2/20 [28:43<4:18:27, 861.5s/it]` in your progress tail gives you the true ETA; recompute the
  budget from it, and fail *fast* on fallback warnings instead of waiting out the cap.
- `ComfyUI did not listen in time` ⇒ check pip's dependency-conflict banner before blaming Comfy.
- Template structure inspection (`definitions.subgraphs`, `widgets_values_named`, link arrays) is
  a 30-second local check that saves a full 10-minute Kaggle round trip.

## Rule for future operators

1. Reproduce the conversion **locally** with a mocked `object_info` (dry-run the submit cell)
   before spending a Kaggle run.
2. `ast.parse` every cell, push, download the log, read the body.
3. One failure per push: fix, push, verify — never bundle three guesses.
