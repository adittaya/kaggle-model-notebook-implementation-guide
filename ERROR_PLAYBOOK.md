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

## Debugging signals that actually solved things

- **Always print the HTTP error body.** v5–v7 were blind; every subsequent fix came from the body.
- `missing_node_type` ⇒ node is frontend-only or the custom pack failed to load → diff against
  `GET /object_info`.
- `KeyError prompt[o_id]` in `validate_inputs` ⇒ dangling link: unflattened subgraph, skipped
  node, or int-vs-string id mismatch.
- `value_not_in_list ... not in ['pixel_space']` ⇒ `models/vae` is **empty** (`pixel_space` is a
  built-in pseudo-VAE, `nodes.py` appends it to every VAE list), so your download never landed.
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
- **v17 multi-modality healthy sequence (expected):** registry prints image/extras sections →
  assets print `OK <bytes>` for 5 video files **+ flux + u2net + RealESRGAN** →
  `[image smoke] build meta {… "modality": "image" …}` → `submitted … node_errors {}` →
  **PNG found** → `[video smoke] submitted` → `active: 2 ranks` → ~195 s/step × 4 → MP4 →
  `SMOKE PASSED` → tunnel → self-test probes all `200` (retry-once, `e.reason` printed on
  retry) → **`TASK SMOKE PASSED`** → READY. **v17 followed exactly this sequence and diverged
  only at the last step** (bg_remove/extract — chain (C) above): when the log matches the
  healthy sequence up to one line, that line *is* the bug. Otherwise start from the first line
  that diverges — a missing flux line means the image schedule didn't load (`H3_SMOKE`), a missing PNG after
  image smoke means the checkpoint never landed (`value_not_in_list` on `ckpt_name`).
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
