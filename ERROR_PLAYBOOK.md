# Error Playbook — How We Fixed Kaggle Model Downloads

This file documents the concrete failure chain encountered on Kaggle while pulling MiniMax H3
weights, and the fixes that actually worked. It's intended as a reusable pattern for any
Kaggle notebook that downloads from Hugging Face.

## Real timeline

| Run | Symptom | Root cause | Fix applied |
|---|---|---|---|
| v2 | `RuntimeError: pinned FL2VA definition not discovered` | Filter matched on model *name* containing "Kaggle" — it didn't | Match by `model_type` prefix (`h3_kaggle_`) |
| v3 | `TypeError: sequence item 0: expected str instance, GenerationError found` | `result.errors` were `GenerationError` objects, never surfaced | Render `e.message` + `e.stage` |
| v4 | `'...url...' is invalid for Model ... [Errno 2] No such file or directory: .../_download_xxx.incomplete` | WanGP's patched `http_get` downloader masked the real error | Bypass it: predownload with `hf_hub_download(local_dir=...)` |
| v5 | `Unable to Download ... cannot find the requested files in the local cache` | HF Hub downloads via `hf_xet` are blocked/unreachable from Kaggle egress | `HF_HUB_DISABLE_XET=1` + `HF_HUB_DISABLE_HF_TRANSFER=1` |
| v6 | **✅ assets fetched** | — | stable stage |

## The canonical fix (used in v6+)

1. Set env **before** any HF import:

```python
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"
```

2. Predownload every asset into the WanGP layout with `hf_hub_download` directly — never rely on WanGP's patched HTTP path for the first fetch:

```python
from huggingface_hub import hf_hub_download
files = [...]
for f in files:
    hf_hub_download(repo_id="DeepBeepMeep/MiniMax-H3", filename=f, local_dir="/tmp/Wan2GP/ckpts")
```

3. Optionally stage from a mounted Kaggle Dataset first (copy is local → no internet at boot):

```python
for base in ["/kaggle/input"]:
    for root,_,files in os.walk(base):
        ...
```

4. Write a truthful manifest of what landed:

```python
for a in files:
    p = CKPT/a
    manifest.append({"file":a, "size_bytes": p.stat().st_size if p.exists() else 0, "present": p.exists()})
```

## Debugging signals we used

- Kawaiting for H3 server... / FastAPI process exited during startup → generation smoke test wrapped the error weirdly → we printed the `GenerationError.message` and `stage`.
- `_download_xxx.incomplete` missing ⇒ downloader patched path was failing; never trust it for first fetch.
- `[Errno 2]` / `cannot find the requested files in the local cache` ⇒ almost always the xet bridge not being reachable from Kaggle; disable xet and it becomes an HTTPS fetch.

## Rule for future operators

*First fetch = `hf_hub_download` with xet disabled, staged out-of-band, into `/tmp/Wan2GP/ckpts` exactly as WanGP expects. Subsequent boots copy from the mounted Dataset or an already-warm `/tmp`. Never let WanGP's patched URL path be the first thing that touches the weights.*
