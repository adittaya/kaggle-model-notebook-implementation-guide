# MiniMax H3 Kaggle Generator — Comfy 2×T4

Deploy a **MiniMax H3** video/audio generator on Kaggle's free `T4 ×2` machine.
This repository is now single-lane: the ComfyUI 0.35-compatible generator
(`minimax_h3_comfy2.ipynb`) with one H3 MultiStream node splitting the H3
transformer across both rigs.

- Fastest path: `kaggle kernels push -p .` (pushes the single Comfy kernel)
- Details on caching / dataset detection / errors:
  - `GUIDE.md`
  - `ERROR_PLAYBOOK.md`
- Task & decision history: `AGENT.md`
- What we found and what we build next: `CONCLUSION.md`
- How to reproduce the same technology for your own models: `BUILD_YOUR_OWN.md`
