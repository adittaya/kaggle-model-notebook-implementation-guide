# GUIDE — Comfy 2×T4 MiniMax H3 Generator

## One-time prerequisites

- Kaggle account with internet enabled and free GPU visible
- The only kernel you should ever need: `adityahalde8777/minimax-h3-comfy-2xt4-generator`
- HF private-repo read via `HF_TOKEN` (Kaggle Secrets) if you mirror assets on HF

## Steps

1. `cd /home/limitlessjourney829/kaggle/minimax-h3-api`
2. `kaggle kernels push -p .`
3. Monitor: `kaggle kernels status adityahalde8777/minimax-h3-comfy-2xt4-generator`
4. Watch logs: `kaggle kernels logs adityahalde8777/minimax-h3-comfy-2xt4-generator`

## What the run does

- clones ComfyUI
- installs the node pack `ComfyUI-H3-MultiStream`
- downloads Comfy-Org/MiniMax-H3 model files:
  `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`,
  `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`,
  `vae/minimax_h3_video_vae_fp16.safetensors`,
  `vae/minimax_h3_audio_vae_fp32.safetensors`
- fetches the official T2V template
- patches it to insert one `H3MultiStream` node (exchange=host, chunks=8)
- queues that patched prompt for the T2V smoke
- Comfy UI stays alive; tunnel URL is the only public endpoint

## Remaining MD set

- `AGENT.md` — track every change
- `CONCLUSION.md` — what we know
- `BUILD_YOUR_OWN.md` — rebuild-your-own map
- `ERROR_PLAYBOOK.md` — what failed during Kaggle downloaders and the H3-MultiStream node installation steps
- `README.md` — at-a-glance
- kernel metadata: `kernel-metadata.json`
