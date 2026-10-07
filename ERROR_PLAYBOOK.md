# Error Playbook — Failure Chains We Hit and Their Fixes

Two chains are documented: **(A) model downloads** (WanGP lane, retired) and
**(B) ComfyUI `/prompt` submissions** (current lane, v3 → v11). Both are reusable patterns.

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
| v15 sync cell: `SKIP upload: cannot determine dataset owner - continuing` (run healthy, write-back silently skipped) | **modern Kaggle notebooks authenticate via a token file** (`KAGGLE_API_V1_TOKEN`), *not* `KAGGLE_USERNAME`/`KAGGLE_KEY`/`~/.kaggle/kaggle.json` — the classic owner check finds nothing; the classic `kaggle` CLI also gets `403` in-kernel | owner = `kagglehub.whoami()` (native in-notebook auth) → env pair → kaggle.json (multi-path incl. `KAGGLE_CONFIG_DIR`) → **embedded kernel owner** (public, build-time from kernel-metadata `id`); upload via `kagglehub.dataset_upload` (creates/versions natively), classic CLI as second fallback; print auth **diagnostics** (booleans only, never secrets) every run so the next log shows which sources exist |

### Debugging signals that actually solved things

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
- **Write-back healthy tail (v16+):** `auth env: KAGGLE_USERNAME=... kaggle.json=...` (booleans)
  → `in-kaggle-notebook: True` → `cache owner: <user> (via kagglehub.whoami)` →
  `cache dataset: <owner>/minimax-h3-model-cache | probe: ...` → `staged N model files` →
  `kagglehub upload OK -> <ref> (<seconds> s)`. Any `SKIP upload:` line means the owner chain
  failed — see the v15 token-file row above.
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

### Rule for future operators

1. Reproduce the conversion **locally** with a mocked `object_info` (dry-run the submit cell)
   before spending a Kaggle run.
2. `ast.parse` every cell, push, download the log, read the body.
3. One failure per push: fix, push, verify — never bundle three guesses.
