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

### Debugging signals that actually solved things

- **Always print the HTTP error body.** v5–v7 were blind; every subsequent fix came from the body.
- `missing_node_type` ⇒ node is frontend-only or the custom pack failed to load → diff against
  `GET /object_info`.
- `KeyError prompt[o_id]` in `validate_inputs` ⇒ dangling link: unflattened subgraph, skipped
  node, or int-vs-string id mismatch.
- `ComfyUI did not listen in time` ⇒ check pip's dependency-conflict banner before blaming Comfy.
- Template structure inspection (`definitions.subgraphs`, `widgets_values_named`, link arrays) is
  a 30-second local check that saves a full 10-minute Kaggle round trip.

### Rule for future operators

1. Reproduce the conversion **locally** with a mocked `object_info` (dry-run the submit cell)
   before spending a Kaggle run.
2. `ast.parse` every cell, push, download the log, read the body.
3. One failure per push: fix, push, verify — never bundle three guesses.
