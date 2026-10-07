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
| 1 | secrets | `/tmp` cache dirs, `HF_HUB_DISABLE_XET=1`, load `HF_TOKEN` (secret, else assembled fallback), `REMBG_HOME=/tmp/ComfyUI/models/rembg` (rembg weights get dataset-cached) |
| 2 | preflight | 2×T4 probe: VRAM free, RAM, `nvidia-smi topo -m` (expect `PHB`, RAM ≈ 31.3 GB) |
| 3 | install | clone ComfyUI + `ComfyUI-H3-MultiStream`, pip install requirements, `pip install rembg[cpu]==2.0.85` (always prints the pip tail), then **v18 numpy/rembg health probe** in a subprocess (`from numpy._core.strings import *; from scipy import ndimage; import rembg`) — on failure print numpy diagnostics (`__path__`, dist-info, `_slice` markers) and **auto-heal** with `pip force-reinstall --no-deps numpy==<ver>`, verify again + **in-kernel** (stale numpy modules dropped) → sets `TASKS_HEALTHY` |
| 4 | **registry** | **live HF catalog**: full `Comfy-Org/MiniMax-H3` tree (39 files: 12 DiT / 3 TE / 3 VAE / 3 LoRA / 4 ControlNet / 10 embeddings) + **flux image tree → `checkpoints/` entries** + **task assets (RealESRGAN + u2net status)** + 100 community repos (and a flux community search), each file status `local`/`cached`/`remote`; scans `/kaggle/input` cache; writes `/tmp/h3_registry.json` — new upstream files/repos auto-list every boot |
| 5 | **settings** | `PRESETS` + `ACTIVE`: `fast_smoke` = turbo ON, **4-step LoRA, 0.4 MP → 864×480**; `quality` = dense 20 steps, 0.98 MP → 1344×768; **`image_smoke`** = Flux 1024², 4 steps, cfg 1, euler/simple; **`H3_SMOKE` schedule** (default `image,video` → one smoke per modality, image first; `H3_SMOKE=video` skips flux); prints the required files per active modality + boot union |
| 6 | assets | **preset/schedule-selective**: mirror `/kaggle/input` hits via symlink (instant), HF `snapshot_download` only the missing files — **video set from `Comfy-Org/MiniMax-H3`, flux into `models/checkpoints/`** — plus task assets (RealESRGAN `.pth` from GitHub releases, rembg `u2net` warm-up), static ffmpeg/ffprobe discovery (`PATH` + `/tmp/bin`), flux template → `/tmp/flux_schnell.json`; **print `OK <bytes>` per file and raise if any is missing**, write `MANIFEST.json` |
| 7 | **convert** | shared prompt builder `build_prompt(cfg)` — **branches on `modality`**: video = settings (instance widgets + ResolutionSelector), optional **first/last-frame conditioning** (data-URL → `LoadImage` nodes 9001/9002 → subgraph inputs), flatten subgraph, arm-check turbo, insert `H3MultiStream`; image = `_build_image` flat converter (CheckpointLoaderSimple / CLIPTextEncode / EmptySD3LatentImage / KSampler / SaveImage, widgets from `widgets_values_named`); used by both the smoke cell and the hub API |
| 8 | start | launch ComfyUI `--preview-method latent2rgb`, wait for `:8188`; start a background `nvidia-smi -l 10` monitor → `/tmp/gpus.log` |
| 9 | **api** | **hub API + reverse proxy on `:8190`** (stdlib `ThreadingHTTPServer`): `/h3api/*` JSON endpoints (incl. **`/tasks` + `/task`** — bg_remove, extract, upscale, video_frames, video_gif, audio_extract, audio_trim, probe) + everything else piped to ComfyUI — HTTP **and** WebSocket (raw bidirectional pipe), chunked bodies, CORS; `H3_API=0` disables (pub then tunnels `:8188` directly) |
| 10 | smoke | loops the **`SMOKE` schedule**: `build_prompt(preset)` → `POST /prompt` with `preview_method: latent2rgb` (prints build meta incl. modality/turbo/node counts) and collects `SMOKE_JOBS=[(modality, prompt_id)]` |
| 11 | wait | polls **every smoke job** (per-modality budget: image 1800 s, video 10800 s), every 120 s print comfy.log tail (tqdm ETA) + new **`[h3ms]` diagnostics** (`[GPUs]` plan, `active: 2 ranks`, per-step times) + **`[gpu]` nvidia-smi util lines**; **early raise on `UNSPLIT`**; fail on execution errors; final assertions are **modality-aware** — video smoke ran ⇒ assert dual-GPU (`active: 2 ranks`), image smoke ran ⇒ assert a PNG output exists; prints `SMOKE PASSED` |
| 12 | pub | cloudflared quick tunnel → `:8190` (the hub), verify **both** `/history` and `/h3api/health` return 200, print READY block (Base URL + hub docs/generate/settings/tasks URLs), then run a **live endpoint self-test through the tunnel** (health/catalog/settings/jobs/outputs/gpus/docs/tasks, proxied `/object_info`, 404 envelope, POST settings — **retry-once + 30 s timeout + `e.reason` printed**, v16 blind `URLError` fixed) **followed by a task smoke through the tunnel**: bg_remove + extract + upscale on the newest PNG (**required when `TASKS_HEALTHY`; bg_remove/extract warn-only if the install cell reported the env unhealthy — upscale always required**), video_frames on the smoke MP4 (warn-only); prints `TASK SMOKE PASSED`; falls back to direct `:8188` if the hub failed to start |
| 13 | **sync** | **Kaggle Dataset write-back** (`H3_CACHE_UPLOAD=auto\|always\|dry\|never`): inventory local models → **200 GB cap auto-skip** + disk guard → owner via **`kagglehub.whoami`** (native in-notebook token auth) with env/kaggle.json/embedded-owner fallbacks + auth diagnostics → remote `MANIFEST.json` probe → manifest-drift check → hardlink-stage (≈0 disk) → **`kagglehub.dataset_upload`** (creates or versions), classic `kaggle` CLI as fallback; **whole cell try/except — a failure never fails the run**. **v16 verified:** `upload OK -> adityahalde8777/minimax-h3-model-cache (530 s)` = 42 GB, first dataset create |

### Model files staged (schedule-selective)

`fast_smoke`/`quality` (video) need exactly these 5:

```
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_int8_convrot.safetensors     ← NOT the fp16 one
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors   ← turbo preset
```

`image_smoke` adds (v17, only when the schedule includes `image`):

```
checkpoints/flux1-schnell-fp8.safetensors             ← 17.24 GB, Comfy-Org/flux1-schnell (repo ROOT file → relocated into checkpoints/)
```

Task assets (always, small): `upscale_models/RealESRGAN_x4plus.pth` (67 MB), rembg `u2net`
(176 MB → `models/rembg/`), static ffmpeg/ffprobe if present.

Resolution order per file: **already local → `/kaggle/input` dataset cache (symlink) → HF download**.
Selecting a different modality **prunes superseded `.safetensors`** (video ⇄ flux).

### Key settings baked into the prompt (from the ACTIVE preset)

- `ResolutionSelector.megapixels = 0.4` → **864×480** (fast_smoke) / `0.98` → 1344×768 (quality)
- turbo path: `PrimitiveBoolean(true)` drives **both** switches → 4-step LoRA + `turbo_steps=4`
- `H3MultiStream`: `exchange="host"`, `exchange_chunks=8`, `second_gpu=-1`,
  `weight_cache=False` (31.3 GB RAM), `vram_block_cache=False`, wired
  after the model switch → `BasicGuider` + `BasicScheduler`
- POST body carries `extra_data: {"preview_method": "latent2rgb"}`
- **No negative prompt exists for H3** (CFG-distilled): prompt + optional first/last frame only

### Hub API behind the same Base URL (v15/v17)

Every response is JSON `{"ok": true, ...}` / `{"ok": false, "error": "..."}`; full HTML reference
at `GET /h3api/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /h3api/health` | API + Comfy liveness, uptime, active preset |
| `GET /h3api/catalog[?refresh=1]` | HF catalog (core + **image** + **extras** + community) with `local/cached/remote`; `refresh` re-queries HF live |
| `GET /h3api/models` | on-disk files + sizes + sources + MANIFEST |
| `GET`/`POST /h3api/settings` | presets (`fast_smoke`, `quality`, **`image_smoke`**) + active settings; POST `{"preset":...}` / `{"set":{...}}` |
| `POST /h3api/select` | switch model: downloads the new set async, **prunes superseded files** (`"prune":false` keeps them) |
| `POST`/`GET /h3api/download` | fetch files now / download status |
| `POST /h3api/generate` | `{prompt, preset?, set?, seed?, first_frame?, last_frame?}` → `202` + `prompt_id`; `preset:"image_smoke"` generates an image (`meta.modality == "image"`) |
| `GET /h3api/jobs` | queue + last 25 history jobs |
| `GET /h3api/outputs` | generated files + `/view` URLs |
| `POST /h3api/free` | unload models from GPU |
| `GET /h3api/gpus` | nvidia-smi snapshot |
| `POST /h3api/import?path=loras/x.safetensors` | raw-bytes model upload (disk-guarded) |
| `GET /h3api/tasks` | the 8 use-case tasks + runtime availability (ffmpeg, rembg, upscale model) |
| `POST /h3api/task` | run a use-case (below) |

#### Use-case tasks (v17) — `POST /h3api/task`

| `task` | Does | Key options |
|---|---|---|
| `bg_remove` | background remover → transparent PNG | `model` (u2net default; u2netp, isnet-general-use, u2net_human_seg, birefnet-*), `alpha_matting` |
| `extract` | element extractor: top-N cutouts + bboxes | `max_elements`, `min_area`, `include_mask` |
| `upscale` | Real-ESRGAN ×4 through Comfy's GPU | — |
| `video_frames` | evenly spaced PNG frames | `count` (1–32), `max_size` |
| `video_gif` | animated preview GIF | `fps`, `width`, `max_duration` |
| `audio_extract` | audio track → wav/mp3 | `format` |
| `audio_trim` | trim a time range | `start`, `duration` |
| `probe` | ffprobe streams/duration/codecs | — |

Input: `image`/`video` as `data:image/...;base64,...` **or** a `<key>_path` (confined to
`/tmp/ComfyUI`, e.g. an MP4 from `/h3api/outputs`). Response: `{ok, outputs:[data URLs],
saved:[names], view:[/view URLs], meta}` — `extract` adds `elements[]` with bboxes; payloads
over 25 MB fall back to saved/view URLs.

Native Comfy (`POST /prompt`, `GET /history`, `GET /ws`, `/view`, the web UI) is proxied unchanged.

## If it fails

- Read the printed HTTP **body** first — see `ERROR_PLAYBOOK.md` for the v3→v12 table.
- A `400 missing_node_type` → something is not in `object_info` (skip UI-only nodes).
- A `400 prompt_outputs_failed_validation` / `KeyError prompt[o_id]` → a dangling link, almost
  always an unflattened subgraph or a non-string link id.
- A `400 value_not_in_list` → the model file never landed (empty folder). `vae_name ... not in
  ['pixel_space']` literally means `models/vae` is empty — check the `OK <bytes>` lines in cell 4;
  `ckpt_name: 'flux1-schnell-fp8.safetensors' not in [...]` means `models/checkpoints` is empty
  (flux missing from the schedule/cache, or pruned by a video-mode `select`).
- Run finished suspiciously fast (~2 min) with no download output → a subprocess exited 0
  without doing anything (this is exactly what `python -m huggingface_hub.cli.download` does).
- **Dual-GPU checks** (search the output): success looks like `[GPUs] dit: 2 rank(s): cuda:0
  (primary), cuda:1` + `[MultiStream] active: 2 ranks cuda:0+cuda:1 …` and
  `[gpu] 0, 100 % … / 1, 100 % …` (both cards ~66 W = T4 ceiling); `running UNSPLIT` /
  `running unsplit on 1 GPU` = single-GPU fallback (the wait cell raises on this **when the video
  smoke ran** — the image smoke legitimately runs single-T4);
  `nvidia-smi` samples printed as `[gpu] 0, 87 %, … MiB, …` / `1, 84 %, …` = live per-card
  utilization.

## Remaining MD set

- `AGENT.md` — task + decision log (mandatory, one line per change)
- `CONCLUSION.md` — what we know, hard numbers
- `BUILD_YOUR_OWN.md` — rebuild-the-same-thing checklist
- `ERROR_PLAYBOOK.md` — every failure chain and its fix
- `README.md` — at-a-glance
- kernel metadata: `kernel-metadata.json`
