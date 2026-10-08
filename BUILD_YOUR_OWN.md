# Build Your Own — Same Technology & Features

Checklist for making your **own** model-server notebook on Kaggle that reproduces the same
features as this repo.

## The stack (current — Comfy lane)

```
Comfy-Org/MiniMax-H3 on HF          Comfy-Org/flux1-schnell (image)   Comfy-Org/MiniMax-Music-3 (music)
        │                                    │                                    │
        ▼                                    ▼                                    ▼
Kaggle kernel (T4×2, internet on, 31.3 GB RAM)
        │
        ├─ ComfyUI (cloned @ main) + pinned requirements
        ├─ ComfyUI-H3-MultiStream custom node  ← the 2-GPU split (video only)
        ├─ official workflow template (subgraph!) → converted to API prompt
        ├─ flat flux template → image API prompt (modality branch in build_prompt)
        ├─ audio_minimax_music_3 template → music API prompt (CORE ComfyUI nodes only:
        │     nodes_minimax_music.py — no custom pack, no MultiStream, single T4)
        ├─ use-case tasks: rembg (bg_remove/extract) + scipy + ffmpeg (video/audio)
        │                  + Real-ESRGAN upscale through Comfy's GPU
        ├─ Comfy /prompt HTTP API on 127.0.0.1:8188
        ├─ hub API + proxy on 127.0.0.1:8190  (/h3api/* JSON + docs + WS passthrough)
        ├─ cloudflared quick tunnel → one public https://*.trycloudflare.com
        │     Comfy UI + native /prompt + /h3api/* share the same Base URL
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
      (`device_count: 2`, `second_gpu=-1`) is not proof. **Healthy pattern (verified on 2×T4):**
      `2 rank(s): cuda:0 (primary), cuda:1` → `active: 2 ranks … heads per rank 28/28` →
      `nvidia-smi` showing both cards at 100 % / ~TDP watts during sampling.
- [ ] **Size wait timeouts from measured throughput, not optimism**: parse the sampler's own
      progress line (`2/20 [28:43<4:18:27, 861.5s/it]` → ETA 4:18) and set the cap with real
      headroom; fail *fast* on fallback warnings instead of waiting out the cap on a doomed run.
- [ ] **Smoke test before READY**: run a real generation; only then print Base URL + endpoint
- [ ] **Preset-driven settings**: one `PRESETS`/`ACTIVE` dict feeds the prompt *and* the download
      list (fast = turbo 4-step @ 0.4 MP, quality = dense 20-step @ 0.98 MP); add `resolution`,
      `duration`, `seed`, `turbo_steps` there — never scatter magic numbers through the converter
- [ ] **Auto-list models from the live source of truth**: query the HF repo tree + community search
      on every boot and print availability (`local`/`cached`/`remote`) — new files/repos appear
      without a code change; cache `/kaggle/input` hits as symlinks (instant mirror), HF download
      only the missing files, then write `MANIFEST.json`
- [ ] **One public Base URL, three planes**: a stdlib `ThreadingHTTPServer` in front of the runtime —
      `/h3api/*` management JSON + an HTML docs page, everything else **piped** to the runtime
      (raw **WebSocket** bidirectional pipe, chunked request bodies, CORS); tunnel *that* port and
      health-check **both** planes through the tunnel before printing READY; a kill-switch env var
      (`H3_API=0`) falls back to tunneling the runtime directly
- [ ] **Settings surface as an API, one builder two callers**: expose preset/override/seed/frame
      conditioning via `POST /generate` built on the **same** `build_prompt()` the smoke cell uses
      — never duplicate the conversion logic in the server
- [ ] **Model lifecycle endpoints**: `select` = download the new set async **and prune superseded
      `.safetensors`** (previous model never lingers); `import` = raw-bytes upload streamed to disk
      with a free-space guard + path-traversal rejection
- [ ] **Dataset write-back that can never kill the run**: stage with **hardlinks** (≈0 extra disk,
      symlink targets preserved), compare a remote `MANIFEST.json` for drift instead of re-uploading
      blindly, **size-cap and auto-skip** (200 GB here), and wrap the entire cell in try/except —
      policy switch (`auto|always|dry|never`) via env var. **Auth matters**: inside a Kaggle
      notebook the classic `kaggle` CLI has no `KAGGLE_USERNAME`/`kaggle.json` — use
      **`kagglehub.whoami()` for the owner and `kagglehub.dataset_upload()` for the push**
      (native token-file auth, creates or versions), keep the CLI as a fallback, embed the
      kernel's own owner (public info) as last resort, and print auth diagnostics (booleans only)
- [ ] **Self-test the public surface before READY**: after the tunnel is up, hit every read-only
      endpoint **through the public URL** (health/catalog/settings/jobs/outputs/gpus/docs, a
      proxied runtime route, a deliberate 404, and a no-op POST) and print status + JSON snippet —
      the log then *proves* the API works end to end, not just that the tunnel exists. **Retry
      each check once with a generous timeout and print the exception's `reason`** (v16: a
      transient tunnel stall on two endpoints was indistinguishable from a slow handler because
      the bare `URLError` was all that got printed)
- [ ] **One low-quality smoke per modality, schedule-driven** (v17/v20): an env var (`H3_SMOKE`,
      default `image,video,music`) defines the smoke order; the smoke cell loops it collecting
      `SMOKE_JOBS=[(modality, prompt_id)]`, the wait cell gives each job its **own budget**
      (image 1800 s, video 10800 s, music 3600 s) and applies **modality-aware assertions** —
      dual-GPU proof only when the multi-GPU (video) modality ran, "output file exists" for the
      fast ones (PNG / FLAC). Don't let a single-T4 image or music lane trip a video-mode
      `active: 2 ranks` assertion. **A music lane sits on CORE ComfyUI nodes** (v20:
      `nodes_minimax_music.py`) — no custom pack, no MultiStream; its template stores inner node
      widgets **positionally**, so promote them to `widgets_values_named` via a **schema-verified
      table** (skip frontend-only slots like `control_after_generate`).
- [ ] **Don't re-run a lane that already passed** (v22, user rule): fingerprint each lane's
      **runtime contract** (resolved config + required model files + sizes) and persist the last-passed
      state at `models/smoke_state.json`; read it back from the cache dataset / workdir next boot and
      **SKIP** lanes whose contract is unchanged (`SMOKE SKIPPED (previously proven)`, still counted in
      `SMOKE PASSED`). Keep escape hatches: `H3_SMOKE_MODE=all` (force full suite after infra changes)
      and `=none` (fast boot, no proof). The fingerprint covers config + files ONLY — a GPU swap with
      identical files skips unless `=all`; that is an accepted, documented trade-off.
- [ ] **`IO.DynamicCombo` inputs in the API prompt are option-key strings** (v21, hard-won): a
      smoke save-node like `SaveAudioAdvanced.format` takes `"flac"` (a plain string), NOT the
      widget dict `{"format":"flac"}` — the executor matches the string against the option list to
      choose sub-inputs and re-nests `{"format": ...}` at call time; a dict silently drops the
      input (`node_errors` stays `{}`, crash at execute with `missing positional 'format'`).
      Sub-inputs of the chosen option (e.g. mp3 → `quality`) are dotted keys (`format.quality`).
      Gate it: run `get_finalized_class_inputs` + `build_nested_inputs` on a local ComfyUI and
      eyeball what `execute()` actually receives before pushing
- [ ] **Ship a pre-push gate script, not just a checklist** (v21): `local_check.py` in-repo rebuilds
      the notebook, `ast`-parses every cell, execs the converter cell against a committed
      `tests/objinfo_fixture.json` (real core schemas + the custom pack's `H3MultiStream`), asserts
      the three smoke lanes structurally, runs the DynamicCombo kwargs deep sim, and — with a local
      ComfyUI running — POSTs each prompt to `/prompt` and asserts the ONLY errors are
      loader-name `value_not_in_list` (empty local model dir). A value-shape bug like v20's `format`
      dict was *invisible to every checklist*; only executing the real executor path catches it.
      **Also gate the notebook FILE itself** (v21/v21b died invisibly — the runner promoted
      nbformat's `MissingIDFieldWarning` to a hard error on cells with NO `id` fields, so the
      notebook never started and produced a 2-byte log + zero artifacts): every generated cell must
      carry a unique `id`, and `nbformat.validate` must pass with **zero warnings** (that check is
      built into the gate).
- [ ] **Separate builder per workflow shape**: a flat template (flux) does not belong inside a
      subgraph flattener — branch `build_prompt(cfg)` on `cfg["modality"]` (video / image / music,
      v20) and keep the verified path untouched; share `load_info()` schema filtering,
      `widgets_values_named`, the Note/MarkdownNote skip, and the **`_flatten(wf, info)` subgraph
      expander** between branches (v20 extracted `_flatten` verbatim from the video path — an
      equivalence test against the old inline flatten must stay byte-identical before you call it
      a refactor)
- [ ] **Use-case tasks beside generation** (v17): `GET /h3api/tasks` (availability: binary
      presence, model files, python deps) + `POST /h3api/task` with one handler per task;
      CPU work in-process (rembg → `scipy.ndimage` connected components for the element
      extractor; ffmpeg for frames/GIF/audio/trim/probe), **GPU work submitted to the runtime
      and polled on `/history`** (Real-ESRGAN via `UpscaleModelLoader`); uniform envelope
      `{ok, outputs:[data URLs], saved:[names], view:[URLs], meta}` with a **payload cap**
      falling back to saved/view URLs; inputs = data URLs or paths **confined to the served
      directory** (400 on escape); 503 with an availability hint when a dep is missing instead
      of a stack trace. Pin the heavy dep (`rembg[cpu]==2.0.85`) and pick a **small default
      model** (u2net 176 MB, not a 1 GB commercial one that can SIGKILL an 8 GB dev box).
- [ ] **Speech-to-text beside generation** (v22, "most of the language support"): use
      **faster-whisper (CTranslate2, CPU int8, no torch)** — not a ComfyUI STT node — so the
      transcription runs on the free CPU and the GPUs stay with Comfy. The SAME Whisper weights
      give ~100 languages on CPU int8 (small: load 3.5 s, 11 s clip → 6.5 s locally). Offer
      model tiers (`tiny/base/small/medium/large-v3`, default `small`), `language: auto|<code>`,
      `task: transcribe|translate`, `txt/srt/vtt` (word timestamps + VAD), and a data-URL / path
      input like the rest of the task engine. Pin deps hard (`faster-whisper==1.2.1 av==13.1.0`
      — fw 1.2.1 needs PyAV 13.x's `metadata_errors` kwarg, removed in av≥19) and keep whisper
      sizes in `models/stt/` so the cache dataset persists them.
- [ ] **Put every runtime asset inside `models/` so the cache picks it up**: rembg weights via
      `REMBG_HOME=/tmp/ComfyUI/models/rembg`, upscaler `.pth` in `models/upscale_models/` —
      anything outside the inventory tree will silently re-download every boot.
- [ ] **Repo-root files need relocation**: some HF repos store the checkpoint at the root, but
      Comfy's combo lists category dirs — download with
      `snapshot_download(..., local_dir=models/<category>)` and track those paths in one shared
      set used by boot, `select`, and prune alike
- [ ] **Probe + heal the CPU-task stack at install time** (v18/v19/v20/v21): a Kaggle boot can't be
      debugged interactively, so after pip installs run the EXACT imports your CPU tasks need in
      a subprocess (`from numpy._core.strings import *; from scipy import ndimage; import rembg`)
      **AND in the kernel**, and on EITHER failure resync numpy with
      `pip install --force-reinstall --no-deps numpy==<version>` — a same-version reinstall
      rewrites every file from one wheel, re-consistent by construction (a same-version resync
      rewrites every file from one wheel — re-consistent by construction). Two traps the runs
      taught us: (1) **v19 lit up why a subprocess-only probe is blind** — a subprocess `pip
      install` (rembg's dep resolve) can swap numpy **in-place** after the kernel already
      pre-loaded it (torch preflight): the disk imports self-consistently (subprocess OK) while
      the kernel's loaded `umath` + new on-disk `strings.py` mix → `cannot import name '_slice'
      from 'numpy._core.umath'`; heal must re-pin to the version the KERNEL loaded
      (`numpy.__version__` in-process), verified locally (2.1.3→2.5.3 swap reproduced, heal
      restores both probes) **and PROVEN in production by v20** (`TASKS_HEALTHY=True` after the
      heal when the kernel-only FAIL hit again). (2) do NOT "fix" an in-kernel import failure by popping the module
      from `sys.modules` and re-importing — same-process re-import of a stem C-extension module
      is broken by itself (`cannot load module more than once per process`; on Kaggle it dies
      inside `multiarray._override___module__` with `'numpy.ufunc' object has no attribute
      '__module__'`), turning one small install hiccup into total in-kernel numpy loss for the
      whole boot. So: **subprocess = the only place on-disk files get re-tested; the kernel only
      reports** (v19 non-destructive verify: ONE attempt, print the FIRST error + traceback tail).
      Survive a permanent failure: gate *secondary-feature* task requirements behind the health
      flag (`TASKS_HEALTHY`) so a bad environment can't block READY — or the write-back cell
      that runs after it — while core generation stays mandatory.
- [ ] **Never discard pip output on rc=0**: a "clean" install can silently shuffle numpy in the
      resolve (v17: a mixed `strings.py`/`umath.py` broke scipy and rembg while `import numpy`
      kept working — the one diagnostic line that would have shown it was thrown away). Always
      log the tail; it's the cheapest lifelong log line you print
- [ ] **Retry-once EVERY client-side request you self-test with — GETs AND POSTs** (v19): GET
      probes retried since v17, but v18's `POST /h3api/settings` (single attempt) and a `POST
      /h3api/task` that died with `Network is unreachable` (tunnel blip) both false-failed a
      healthy stack. One shared `_post` helper (retry, `e.reason` log, raise only after both
      attempts) for all self-test POSTs keeps task smoke honest
- [ ] **Keep the notebook generator inside the repo (Phase E)**: `/tmp` may be wiped by a host
      restart (it happened — the generator and dry-run harness died with it). Commit the build
      script; embed each cell as a raw `r'''…'''` string (regular strings eat the `\n`/`\.`
      sequences living in cell sources); a wrong embedding was caught by diffing the rebuilt
      notebook against `git show HEAD:….ipynb` — make that diff part of the gate after any
      structural change
- [ ] **Double escapes meant for the notebook**: cell code with `"\n"` / `b"\r\n"` sits inside the
      build script's triple-quoted string, which consumes the backslashes — write `\\n` there, and
      `ast.parse` every **generated** cell (the build script parsing cleanly proves nothing)
- [ ] **Local dry-run harness**: execute the notebook cells against a local CPU ComfyUI (start it
      with `--cpu` on CPU-only torch), confirm `/prompt` returns a `prompt_id`, **and exercise every
      HTTP endpoint you added** (happy paths, error codes, WS `101` handshake, chunked body, and the
      write-back cell's `dry` policy) before pushing. **Never let big weights download locally**:
      write 12-byte placeholders for every expected `.safetensors` at harness boot — and re-create
      them right before any test that runs *after* your own prune/select tests (Comfy re-scans its
      combo lists, so a pruned placeholder genuinely disappears from validation)
- [ ] **Quality-first defaults**: turbo/TeaCache/Spectrum/FBC off (this template's switches default to dense 20-step)
- [ ] **Public tunnel**: `cloudflared tunnel --url http://127.0.0.1:<hub-port>`, health-check the
      hub health endpoint **and** the runtime's `/history` through it (fall back to `:8188` direct
      if the hub failed to start)
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

1. Upload the assets once — **or let the notebook do it**: the `sync` cell ran first in v16 and
   **created** `adityahalde8777/minimax-h3-model-cache` itself via `kagglehub.dataset_upload`
   (42 GB in 530 s, no manual `kaggle datasets create` needed)
2. In the assets cell, **first** copy matching files out of `/kaggle/input/` into
   `/tmp/ComfyUI/models/<subdir>/`, keeping the exact directory layout ComfyUI expects
   (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`, `checkpoints/`,
   `upscale_models/`, `rembg/` — v17 added image + task assets to the same inventory)
3. Then run `snapshot_download(...)` only for files **still missing**, and print the size of
   every file afterwards (exit code 0 proves nothing — a silent no-op "succeeds" too)
4. Write a `MANIFEST.json` of what actually landed (name, size, present)
5. **v15/v16 automates the upload half**: a `sync` cell inventories local models, checks the
   remote `MANIFEST.json` for drift, hardlink-stages and pushes back — **primary path
   `kagglehub.dataset_upload`** (native in-notebook auth; owner from `kagglehub.whoami()`),
   classic `kaggle datasets create/version` as fallback — policy via
   `H3_CACHE_UPLOAD=auto|always|dry|never`, **200 GB cap auto-skips**, and *any* failure
   prints and continues (the run never dies on it).
   CLI gotchas (fallback path): `datasets version` has **no `-d`** — it reads
   `dataset-metadata.json` (`{title, id, licenses}`) from the `-p` folder; `datasets list`
   CSV flag is `-v`; existence probe = `kaggle datasets metadata <owner>/<slug>`.
   One-time manual step: attach the created dataset to the kernel (Add Data) so `/kaggle/input`
   holds it — next boots then mirror it via symlink instead of downloading.
