# Build Your Own — Same Technology & Features

Checklist for making your **own** model-server notebook on Kaggle that reproduces the same
features as this repo.

## The stack (current — Comfy lane)

```
Comfy-Org/MiniMax-H3 on HF
        │
        ▼
Kaggle kernel (T4×2, internet on, 31.3 GB RAM)
        │
        ├─ ComfyUI (cloned @ main) + pinned requirements
        ├─ ComfyUI-H3-MultiStream custom node  ← the 2-GPU split
        ├─ official workflow template (subgraph!) → converted to API prompt
        ├─ Comfy /prompt HTTP API on 127.0.0.1:8188
        ├─ cloudflared quick tunnel → public https://*.trycloudflare.com
        └─ everything in /tmp (session-only)
```

### Retired stack (WanGP lane, deleted — kept for reference)

```
WanGP pinned commit → local finetune JSONs → FastAPI headless server
→ cloudflared quick tunnel → polling job API
```

## Feature checklist

- [ ] **Pinned upstream**: clone the runtime and `git checkout <commit>` (or pin pip versions)
- [ ] **Auth from secrets**: never hardcode tokens; read `HF_TOKEN` / `KAGGLE_KEY` from Kaggle Secrets
- [ ] **Cache hygiene**: `HF_HOME=/tmp/hf`, `TORCH_HOME=/tmp/torch`, `XDG_CACHE_HOME=/tmp/xdg_cache`
- [ ] **Download safety**: `HF_HUB_DISABLE_XET=1`, `HF_HUB_DISABLE_HF_TRANSFER=1`, and **never**
      `pip install -U huggingface_hub` (breaks the runtime's pinned deps)
- [ ] **Asset names == widget names**: the files you download must match the workflow's combo
      values exactly (`..._int8_convrot.safetensors` vs `..._fp16.safetensors` matters)
- [ ] **API-format prompts**: convert saved workflow → `{id:{class_type,inputs}}`, link ids as
      **strings**, drop UI-only nodes (verify against `GET /object_info`), **flatten subgraphs**
- [ ] **Wire the acceleration node** where the docs say (here: after model patches →
      `BasicGuider` + `BasicScheduler`)
- [ ] **Send `extra_data.preview_method`** — API clients must set it explicitly
- [ ] **Verify side effects, not exit codes**: print `OK <bytes>` for every downloaded file and
      raise on any miss. Never `python -m <pkg>.cli.<cmd>` without checking the module has an
      `if __name__ == "__main__"` guard — `huggingface_hub.cli.download` has none, so it exits 0
      having downloaded nothing. Prefer in-process `snapshot_download()`
- [ ] **Pass every linked input through**, even if its name is absent from the static
      `object_info` schema (autogrow slots like `values.a` only exist after finalization); only
      *widgets* get filtered against the schema
- [ ] **Prove multi-GPU execution at runtime**: start
      `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 10 > /tmp/gpus.log`
      alongside the server and print its tail in your wait loop; after the job, **assert** the
      pack's own evidence (here `[MultiStream] active: 2 ranks`) is in the server log and
      **raise** on any single-GPU fallback warning (`UNSPLIT`). Intent in the submitted config
      (`device_count: 2`, `second_gpu=-1`) is not proof.
- [ ] **Size wait timeouts from measured throughput, not optimism**: parse the sampler's own
      progress line (`2/20 [28:43<4:18:27, 861.5s/it]` → ETA 4:18) and set the cap with real
      headroom; fail *fast* on fallback warnings instead of waiting out the cap on a doomed run.
- [ ] **Smoke test before READY**: run a real generation; only then print Base URL + endpoint
- [ ] **Quality-first defaults**: turbo/TeaCache/Spectrum/FBC off (this template's switches default to dense 20-step)
- [ ] **Public tunnel**: `cloudflared tunnel --url http://127.0.0.1:8188`, health-check `/history`
- [ ] **Guarded cleanup**: only act on `H3_SHUTDOWN=1`

## 2×T4 guide (Kaggle)

Kaggle `machine_shape: NvidiaTeslaT4` = 2 × T4 16 GB, **31.3 GB system RAM**, PCIe, no NVLink.

- Topology: `nvidia-smi topo -m` → `GPU0 ↔ GPU1 = PHB`
- Measured exchange: host ≈ 10 GB/s, p2p ≈ 9.3 GB/s ⇒ **`exchange="host"` + `exchange_chunks=8`**
- 31.3 GB RAM ⇒ **all pinned weight caches off** (DiT cache alone wants 18 GiB)
- Single-job multi-GPU = `ComfyUI-H3-MultiStream`; two independent workers = throughput fallback

## Prompt conversion recipe (the hard-won part)

```python
info  = json.loads(urllib.request.urlopen(BASE + "/object_info").read())   # registered nodes
valid = set(info[ct]["input"]["required"]) | set(info[ct]["input"]["optional"])

# saved workflow: nodes have `inputs[].link` + `links[]` + `widgets_values_named`
# API prompt:     {str(id): {"class_type": t, "inputs": {...}}}
for inp in node["inputs"]:
    if inp["name"] in valid and inp.get("link") is not None:
        src = link_lookup[inp["link"]]                 # [link_id, src, slot, dst, slot, type]
        inputs[inp["name"]] = [str(src[1]), src[2]]    # STRING node id!
for name in valid:                                     # widgets, schema-filtered
    if name not in inputs and name in widgets_named:
        inputs[name] = widgets_named[name]             # drops frontend-only keys

# subgraph instance (type = UUID in workflow["definitions"]["subgraphs"]):
#   -> expand inner nodes with fresh unique ids
#   -> subgraph INPUT  port: instance widget value / external link / omit if empty
#   -> subgraph OUTPUT port: the inner node feeding that port
```

## Your repository structure

```
your-kernel-repo/
├── README.md                  # quick intro
├── GUIDE.md                   # deployment guide
├── BUILD_YOUR_OWN.md          # this file
├── AGENT.md                   # every task/decision line MUST be logged
├── CONCLUSION.md              # what we learned and why
├── ERROR_PLAYBOOK.md          # every failure chain + fix
├── kernel-metadata.json       # machine_shape: NvidiaTeslaT4, internet on
└── your_notebook.ipynb        # generated by your build script
```

## Mandatory dev discipline

1. Every run → append to `AGENT.md` task log + decision log
2. Every configuration result → append to `CONCLUSION.md`
3. **After any significant change → update *all* MD files in the repo**
4. `git add -A && git commit && git push` after every significant change
5. Regenerate the notebook from a script and `ast`-validate every cell before pushing
6. Never commit credentials; rotate any leaked PAT/token

## Minimal viable command set

```bash
kaggle kernels push -p .
kaggle kernels status OWNER/kernel
kaggle kernels logs OWNER/kernel > log.json
kaggle kernels delete -y OWNER/kernel/old

git add -A; git commit -m "progress: ..."; git push
```

---

## Persistent model caching across sessions (Kaggle Dataset)

Needed step — skip the HF download on every boot by shipping the weights in a Kaggle Dataset and
letting the notebook auto-detect the mount:

1. Upload the assets once: `kaggle datasets create -p <folder with the H3 files>`
2. In the assets cell, **first** copy matching files out of `/kaggle/input/` into
   `/tmp/ComfyUI/models/<subdir>/`, keeping the exact directory layout ComfyUI expects
   (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`)
3. Then run `snapshot_download(...)` only for files **still missing**, and print the size of
   every file afterwards (exit code 0 proves nothing — a silent no-op "succeeds" too)
4. Write a `MANIFEST.json` of what actually landed (name, size, present)
