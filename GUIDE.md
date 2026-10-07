# GUIDE — Comfy 2×T4 MiniMax H3 Generator

## One-time prerequisites

- Kaggle account with internet enabled and free GPU visible
- The only kernel you should ever need: `adityahalde8777/minimax-h3-comfy-2xt4-generator`
- HF read access to `Comfy-Org/MiniMax-H3` (public) — `HF_TOKEN` secret only if you mirror it
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
| 1 | secrets | `/tmp` cache dirs, `HF_HUB_DISABLE_XET=1`, load `HF_TOKEN` |
| 2 | preflight | 2×T4 probe: VRAM free, RAM, `nvidia-smi topo -m` (expect `PHB`, RAM ≈ 31.3 GB) |
| 3 | install | clone ComfyUI + `ComfyUI-H3-MultiStream`, pip install requirements |
| 4 | assets | `snapshot_download` 5 files from `Comfy-Org/MiniMax-H3` into `/tmp/ComfyUI/models`, **print `OK <bytes>` per file and raise if any is missing** + fetch the T2V template |
| 5 | inspect | report top-level nodes / subgraph names |
| 6 | start | launch ComfyUI `--preview-method latent2rgb`, wait for `:8188`; start a background `nvidia-smi -l 10` monitor → `/tmp/gpus.log` |
| 7 | submit | **convert saved workflow → API prompt** (flatten subgraph), insert `H3MultiStream`, `POST /prompt` |
| 8 | wait | poll `/history/<id>` up to **6.5 h**, every 120 s print comfy.log tail (tqdm ETA) + new **`[h3ms]` diagnostics** (`[GPUs]` plan, `active: 2 ranks`, per-step times) + **`[gpu]` nvidia-smi util lines**; **early raise on `UNSPLIT`**; fail on execution errors; final **assert dual-GPU** (raise unless `active: 2 ranks`) |
| 9 | pub | cloudflared quick tunnel, verify `/history` returns 200, print READY block |

### Model files staged (exact widget names from the official template)

```
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_int8_convrot.safetensors     ← NOT the fp16 one
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

### Key settings baked into the prompt

- `ResolutionSelector.megapixels = 0.98` → **1344×768**
- `H3MultiStream`: `exchange="host"`, `exchange_chunks=8`, `second_gpu=-1`,
  `weight_cache=False` (31.3 GB RAM), `vram_block_cache=False`, wired
  after the model switch → `BasicGuider` + `BasicScheduler`
- dense run (template default: turbo lora switch = false, 20 steps)
- POST body carries `extra_data: {"preview_method": "latent2rgb"}`

## If it fails

- Read the printed HTTP **body** first — see `ERROR_PLAYBOOK.md` for the v3→v12 table.
- A `400 missing_node_type` → something is not in `object_info` (skip UI-only nodes).
- A `400 prompt_outputs_failed_validation` / `KeyError prompt[o_id]` → a dangling link, almost
  always an unflattened subgraph or a non-string link id.
- A `400 value_not_in_list` → the model file never landed (empty folder). `vae_name ... not in
  ['pixel_space']` literally means `models/vae` is empty — check the `OK <bytes>` lines in cell 4.
- Run finished suspiciously fast (~2 min) with no download output → a subprocess exited 0
  without doing anything (this is exactly what `python -m huggingface_hub.cli.download` does).
- **Dual-GPU checks** (search the output): `[MultiStream] active: 2 ranks` or `[GPUs] dit: …
  cuda:0, cuda:1` = both T4s in the split; `running UNSPLIT` / `running unsplit on 1 GPU` =
  single-GPU fallback (the wait cell now raises on this); `nvidia-smi` samples printed as
  `[gpu] 0, 87 %, … MiB, …` / `1, 84 %, …` = live per-card utilization.

## Remaining MD set

- `AGENT.md` — task + decision log (mandatory, one line per change)
- `CONCLUSION.md` — what we know, hard numbers
- `BUILD_YOUR_OWN.md` — rebuild-the-same-thing checklist
- `ERROR_PLAYBOOK.md` — every failure chain and its fix
- `README.md` — at-a-glance
- kernel metadata: `kernel-metadata.json`
