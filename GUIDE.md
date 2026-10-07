# GUIDE — Comfy 2×T4 MiniMax H3 Generator

## One-time prerequisites

- Kaggle account with internet enabled and free GPU visible
- The only kernel you should ever need: `adityahalde8777/minimax-h3-comfy-2xt4-generator`
- HF token (`HF_TOKEN` Kaggle secret; assembled fallback embedded) for rate-limit-free downloads
- Repo is generated: edit `/tmp/opencode/build_full.py`, **never** the `.ipynb` by hand

## Steps

1. `cd /home/limitlessjourney829/kaggle/minimax-h3-api`
2. `python3 /tmp/opencode/build_full.py`  ← regenerates `minimax_h3_full_comfy.ipynb`
3. Validate: every code cell must pass `ast.parse`
4. `kaggle kernels push -p .`
5. Monitor: `kaggle kernels status adityahalde8777/minimax-h3-comfy-2xt4-generator`
6. Logs: `kaggle kernels logs adityahalde8777/minimax-h3-comfy-2xt4-generator > log.json`
7. `git add -A && git commit -m "..." && git push`
8. **Update all MD files** (`AGENT.md`, `CONCLUSION.md`, `GUIDE.md`, `BUILD_YOUR_OWN.md`,
   `ERROR_PLAYBOOK.md`, `README.md`)

## What the run does (cell by cell)

| # | Cell | What happens |
|---|---|---|
| 1 | secrets | `/tmp` cache dirs, `HF_HUB_DISABLE_XET=1`, load `HF_TOKEN` (secret, else assembled fallback) |
| 2 | preflight | 2×T4 probe: VRAM free, RAM, `nvidia-smi topo -m` (expect `PHB`, RAM ≈ 31.3 GB) |
| 3 | install | clone ComfyUI + `ComfyUI-H3-MultiStream`, pip install requirements |
| 4 | **registry** | **live HF catalog**: full `Comfy-Org/MiniMax-H3` tree (39 files: 12 DiT / 3 TE / 3 VAE / 3 LoRA / 4 ControlNet / 10 embeddings) + 100 community repos, each file status `local`/`cached`/`remote`; scans `/kaggle/input` cache; writes `/tmp/h3_registry.json` — new upstream files/repos auto-list every boot |
| 5 | **settings** | `PRESETS` + `ACTIVE`: `fast_smoke` = turbo ON, **4-step LoRA, 0.4 MP → 864×480**; `quality` = dense 20 steps, 0.98 MP → 1344×768; prints the exact 5 required files |
| 6 | assets | **preset-selective**: mirror `/kaggle/input` hits via symlink (instant), HF `snapshot_download` only the missing files, **print `OK <bytes>` per file and raise if any is missing**, write `MANIFEST.json` + fetch the T2V template |
| 7 | start | launch ComfyUI `--preview-method latent2rgb`, wait for `:8188`; start a background `nvidia-smi -l 10` monitor → `/tmp/gpus.log` |
| 8 | submit | **apply ACTIVE settings** to the workflow (instance widgets + ResolutionSelector), convert saved workflow → API prompt (flatten subgraph), arm-check turbo (`PrimitiveBoolean → steps 4 + LoRA`), insert `H3MultiStream`, `POST /prompt` |
| 9 | wait | poll `/history/<id>` up to **3 h**, every 120 s print comfy.log tail (tqdm ETA) + new **`[h3ms]` diagnostics** (`[GPUs]` plan, `active: 2 ranks`, per-step times) + **`[gpu]` nvidia-smi util lines**; **early raise on `UNSPLIT`**; fail on execution errors; final **assert dual-GPU** (raise unless `active: 2 ranks`) |
| 10 | pub | cloudflared quick tunnel, verify `/history` returns 200, print READY block |

### Model files staged (preset-selective — `fast_smoke` needs exactly these 5)

```
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_int8_convrot.safetensors     ← NOT the fp16 one
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors   ← turbo preset
```

Resolution order per file: **already local → `/kaggle/input` dataset cache (symlink) → HF download**.

### Key settings baked into the prompt (from the ACTIVE preset)

- `ResolutionSelector.megapixels = 0.4` → **864×480** (fast_smoke) / `0.98` → 1344×768 (quality)
- turbo path: `PrimitiveBoolean(true)` drives **both** switches → 4-step LoRA + `turbo_steps=4`
- `H3MultiStream`: `exchange="host"`, `exchange_chunks=8`, `second_gpu=-1`,
  `weight_cache=False` (31.3 GB RAM), `vram_block_cache=False`, wired
  after the model switch → `BasicGuider` + `BasicScheduler`
- POST body carries `extra_data: {"preview_method": "latent2rgb"}`
- **No negative prompt exists for H3** (CFG-distilled): prompt + optional first/last frame only

## If it fails

- Read the printed HTTP **body** first — see `ERROR_PLAYBOOK.md` for the v3→v12 table.
- A `400 missing_node_type` → something is not in `object_info` (skip UI-only nodes).
- A `400 prompt_outputs_failed_validation` / `KeyError prompt[o_id]` → a dangling link, almost
  always an unflattened subgraph or a non-string link id.
- A `400 value_not_in_list` → the model file never landed (empty folder). `vae_name ... not in
  ['pixel_space']` literally means `models/vae` is empty — check the `OK <bytes>` lines in cell 4.
- Run finished suspiciously fast (~2 min) with no download output → a subprocess exited 0
  without doing anything (this is exactly what `python -m huggingface_hub.cli.download` does).
- **Dual-GPU checks** (search the output): success looks like `[GPUs] dit: 2 rank(s): cuda:0
  (primary), cuda:1` + `[MultiStream] active: 2 ranks cuda:0+cuda:1 …` and
  `[gpu] 0, 100 % … / 1, 100 % …` (both cards ~66 W = T4 ceiling); `running UNSPLIT` /
  `running unsplit on 1 GPU` = single-GPU fallback (the wait cell raises on this);
  `nvidia-smi` samples printed as `[gpu] 0, 87 %, … MiB, …` / `1, 84 %, …` = live per-card
  utilization.

## Remaining MD set

- `AGENT.md` — task + decision log (mandatory, one line per change)
- `CONCLUSION.md` — what we know, hard numbers
- `BUILD_YOUR_OWN.md` — rebuild-the-same-thing checklist
- `ERROR_PLAYBOOK.md` — every failure chain and its fix
- `README.md` — at-a-glance
- kernel metadata: `kernel-metadata.json`
