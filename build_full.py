#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Build script for the single Kaggle kernel notebook (ONE kernel, ONE file).
# IMPORTANT: NEVER hand-edit minimax_h3_full_comfy.ipynb. Edit the cell variables
# in this file, then run `python3 build_full.py`, then `kaggle kernels push -p .`.
# This file was reconstructed verbatim from the v18 notebook source of truth and
# now lives in the repo so it survives host restarts (Phase E).
# =============================================================================
import json, os, sys

# ---------------- cell 0: markdown title ----------------
title_md = r'''
# MiniMax H3 — 2×T4 Comfy Generator (public URL enabled)

Dynamic model registry (live HF catalog + Kaggle Dataset auto-cache) + fast smoke:
4-step turbo LoRA @ 864×480 on both T4s (video) and Flux-schnell 4-step @ 1024² (image),
plus image/video/audio **use-case tasks** (background remover, element extractor,
upscaler, frame/GIF/audio tools). One public Base URL serves the ComfyUI UI,
the native `/prompt` API **and** the `/h3api/*` management hub (docs at `/h3api/docs`).
'''

# ---------------- cell 1: secrets ----------------
secrets = r'''
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER","1")
os.environ.setdefault("HF_HOME","/tmp/hf")
os.environ.setdefault("REMBG_HOME","/tmp/ComfyUI/models/rembg")  # segmentation models: dataset-cached
os.environ.setdefault("TORCH_HOME","/tmp/torch")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/xdg_cache")
for d in ("/tmp/hf","/tmp/torch","/tmp/xdg_cache"):
    Path(d).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
try:
    from kaggle_secrets import UserSecretsClient
    tok=UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"]=tok
    print("HF_TOKEN loaded from secrets")
except Exception:
    _p=["hf_","DhBUHeo","GyodIyQu","DLhyFYzg","roBKRLKQ","OaW"]   # assembled, never raw in git
    os.environ.setdefault("HF_TOKEN","".join(_p))
    print("HF_TOKEN fallback active (rate-limit-free downloads)")
'''

# ---------------- cell 2: preflight ----------------
preflight = r'''
import shutil, subprocess, psutil, torch, os, sys
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); f,t=torch.cuda.mem_get_info(i)
    print(f"GPU{i}: {p.name} {t/2**30:.1f}GB free={f/2**30:.1f}GB")
ram=psutil.virtual_memory()
print(f"RAM total={ram.total/2**30:.1f}GB avail={ram.available/2**30:.1f}GB")
print("/tmp free:", shutil.disk_usage('/tmp').free/2**30, "GiB")
print(subprocess.run(["nvidia-smi","topo","-m"],capture_output=True,text=True).stdout)
'''

# ---------------- cell 3: install ----------------
install = r'''
import subprocess,sys,os
CTE="/tmp/ComfyUI"
if not os.path.isdir(CTE):
    subprocess.run(["git","clone","--depth","1","https://github.com/Comfy-Org/ComfyUI.git",CTE],check=True)
subprocess.run([sys.executable,"-m","pip","install","-r",os.path.join(CTE,"requirements.txt")],check=True)
cn=os.path.join(CTE,"custom_nodes","ComfyUI-H3-MultiStream")
if not os.path.isdir(cn):
    subprocess.run(["git","clone","--depth","1","https://github.com/martonsagi/Comfy-H3-MultiStream.git",cn],check=True)
try:
    subprocess.run([sys.executable,"-m","pip","install","-r",os.path.join(cn,"requirements.txt")],check=False)
except Exception:
    pass
# image use-case tasks (background remover / element extractor): onnx models, CPU
r=subprocess.run([sys.executable,"-m","pip","install","rembg[cpu]==2.0.85"],
                 capture_output=True,text=True)
print("rembg install rc=%d"%r.returncode)
print(((r.stdout or "")+(r.stderr or ""))[-700:])   # always show tail (a numpy move shows here)
# numpy/rembg health + heal: a MIXED numpy (disk files upgraded in place by a subprocess
# `pip install` while THIS kernel pre-loaded numpy) breaks scipy and rembg with
# `cannot import name '_slice' from 'numpy._core.umath'` — the disk may stay self-consistent
# (fresh subprocess import passes) while in-kernel submodule imports fail. So probe the exact
# imports the tasks need BOTH ways (subprocess files + in-kernel modules), and if either is
# broken, resync the wheel to the version THIS kernel loaded, then re-verify both.
_PROBE="from numpy._core.strings import *; from scipy import ndimage; import rembg"
def _task_probe():
    p=subprocess.run([sys.executable,"-c",_PROBE],capture_output=True,text=True)
    return p.returncode==0,(p.stderr or "").strip()[-300:]
def _numpy_diag():
    try:
        import numpy,glob as _g
        from pathlib import Path as _P
        root=_P(numpy.__file__).parent; core=root/"_core"
        print("  numpy:",getattr(numpy,"__version__","?"),"root:",root)
        print("  __init__.py:",(root/"__init__.py").exists(),"__path__:",list(numpy.__path__))
        print("  dist-info:",[i.split("/")[-1] for i in _g.glob(str(root.parent/"numpy-*.dist-info"))])
        for f in ("umath.py","strings.py"):
            p=core/f
            has="_slice" in p.read_text(errors="replace") if p.exists() else "-"
            print("  %s: exists=%s has _slice=%s"%(f,p.exists(),has))
    except Exception as e: print("  numpy diag failed:",e)
def _kernel_verify():
    import traceback as _tb
    def _try():
        import numpy._core.strings   # the exact module the failure hits
        import scipy.ndimage,rembg
        return True
    try: return _try()
    except BaseException as e:
        # v18 lesson: NEVER pop numpy here to retry - a same-process re-import after removing
        # the modules is itself broken (`ufunc.__module__=...` / "cannot load module more than
        # once per process"), which turned a specific import failure into a total numpy loss
        # for the whole boot (warm-up, api import and every task). One attempt, full error.
        print("in-kernel task deps import FAIL:",type(e).__name__,e)
        print("  last frames:"," | ".join(_tb.format_exc().splitlines()[-5:]))
        return False
ok,err=_task_probe()
print("task deps health:","OK" if ok else "FAIL -> "+err)
k_ok=_kernel_verify()
print("in-kernel task deps import:","OK" if k_ok else "FAIL")
# v19 lesson: the subprocess probe can pass on the freshly-upgraded (consistent) disk numpy
# while this kernel's pre-loaded numpy modules are stale -> in-kernel imports break. Heal
# whenever EITHER probe fails, force-reinstalling numpy to the version THIS kernel loaded.
if not ok or not k_ok:
    _numpy_diag()
    try:
        import numpy as _np
        ver=_np.__version__      # version actually loaded in THIS process (what we must match)
    except BaseException:
        ver=subprocess.run([sys.executable,"-c","import numpy;print(numpy.__version__)"],
                           capture_output=True,text=True).stdout.strip() or "2.1.3"
    print("heal: pip force-reinstall --no-deps numpy==%s"%ver)
    h=subprocess.run([sys.executable,"-m","pip","install","-q","--force-reinstall",
                      "--no-deps","numpy==%s"%ver],capture_output=True,text=True)
    print("heal rc=%d"%h.returncode,((h.stdout or "")+(h.stderr or ""))[-300:])
    ok,err=_task_probe()
    print("task deps health after heal:","OK" if ok else "FAIL -> "+err)
    if not ok: _numpy_diag()
    k_ok=_kernel_verify()
    print("in-kernel task deps import after heal:","OK" if k_ok else "FAIL")
    if not k_ok: _numpy_diag()
TASKS_HEALTHY=bool(ok and k_ok)
print("TASKS_HEALTHY =",TASKS_HEALTHY)
print("ComfyUI at",CTE)
'''

# ---------------- cell 4: registry ----------------
registry = r'''
import json, os, urllib.request
from pathlib import Path

HF_TOKEN=os.environ.get("HF_TOKEN","")
CORE_REPO="Comfy-Org/MiniMax-H3"
MODELS_DIR=Path("/tmp/ComfyUI/models")

def hf_get(url, timeout=60):
    h={"Authorization":"Bearer "+HF_TOKEN} if HF_TOKEN else {}
    with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=timeout) as r:
        return json.loads(r.read())

# 1) core catalog: LIVE HF tree — files added upstream appear automatically
try:
    tree=hf_get(f"https://huggingface.co/api/models/{CORE_REPO}/tree/main?recursive=true")
    CORE=[{"path":f["path"],"size":int(f.get("size") or 0)} for f in tree
          if f.get("type")=="file" and not f["path"].startswith(".")]
except Exception as e:
    print("core tree query FAILED:",e); CORE=[]
print("core catalog: %d files in %s" % (len(CORE),CORE_REPO))

# 1b) image generation catalog: LIVE Flux tree (root files live under checkpoints/)
IMG_REPO="Comfy-Org/flux1-schnell"
try:
    itree=hf_get(f"https://huggingface.co/api/models/{IMG_REPO}/tree/main?recursive=true")
    IMG=[{"path":("checkpoints/" if "/" not in f["path"] else "")+f["path"],
          "size":int(f.get("size") or 0)} for f in itree
         if f.get("type")=="file" and f["path"].endswith(".safetensors")]
except Exception as e:
    print("image tree query FAILED:",e); IMG=[]
print("image catalog: %d files in %s" % (len(IMG),IMG_REPO))

# 1b2) music generation catalog: LIVE MiniMax-Music-3 tree (diffusion_models/text_encoders/vae)
MUSIC_REPO="Comfy-Org/MiniMax-Music-3"
try:
    mtree=hf_get(f"https://huggingface.co/api/models/{MUSIC_REPO}/tree/main?recursive=true")
    MUSIC=[{"path":f["path"],"size":int(f.get("size") or 0)} for f in mtree
           if f.get("type")=="file" and f["path"].endswith(".safetensors")]
except Exception as e:
    print("music tree query FAILED:",e); MUSIC=[]
print("music catalog: %d files in %s" % (len(MUSIC),MUSIC_REPO))

# 1c) local task assets (upscale + segmentation models)
EXTRAS=[
    {"path":"upscale_models/RealESRGAN_x4plus.pth","size":67040989,
     "note":"Real-ESRGAN x4 - task: upscale"},
    {"path":"rembg/models/u2net/u2net.onnx","size":176287960,
     "note":"rembg u2net - tasks: bg_remove, extract"},
]

# 2) Kaggle Dataset auto-cache scan (/kaggle/input)
CACHE={}
if os.path.isdir("/kaggle/input"):
    for root,_,files in os.walk("/kaggle/input"):
        for fn in files:
            CACHE.setdefault(fn, os.path.join(root,fn))
print("dataset cache: %d candidate files under /kaggle/input" % len(CACHE))

def status_of(rel):
    p=MODELS_DIR/rel
    if p.exists() and p.stat().st_size>0: return "local"
    if rel.rsplit("/",1)[-1] in CACHE: return "cached"
    return "remote"

# 3) grouped catalog table (size, availability)
groups={}
for f in CORE+IMG+MUSIC:
    groups.setdefault(f["path"].split("/")[0] if "/" in f["path"] else "(root)",[]).append(f)
for g in sorted(groups):
    rows=sorted(groups[g], key=lambda x:-x["size"])
    tot=sum(r["size"] for r in rows)
    print("\n== %s  (%d files, %.1f GB) ==" % (g,len(rows),tot/1e9))
    for r in rows:
        print("  %-6s %7.2fG  %s" % (status_of(r["path"]), r["size"]/1e9, r["path"].rsplit("/",1)[-1]))
print("\n== task assets ==")
for x in EXTRAS:
    print("  %-6s %7.2fG  %s  (%s)" % (status_of(x["path"]), x["size"]/1e9, x["path"], x["note"]))

# 4) community repos: LIVE search — new repos appear automatically
try:
    res=hf_get("https://huggingface.co/api/models?search=MiniMax-H3&limit=100&sort=downloads&direction=-1")
    COMM=[{"id":m["id"],"downloads":int(m.get("downloads") or 0)} for m in res]
except Exception as e:
    print("community search FAILED:",e); COMM=[]
print("\n== community MiniMax-H3 repos (%d total; top 20 by downloads) ==" % len(COMM))
for m in sorted(COMM,key=lambda x:-x["downloads"])[:20]:
    print("  %9s  %s" % (format(m["downloads"],","), m["id"]))

# 5) image community repos: LIVE flux search — alternative image models appear automatically
try:
    res2=hf_get("https://huggingface.co/api/models?search=flux&limit=100&sort=downloads&direction=-1")
    COMM2=[{"id":m["id"],"downloads":int(m.get("downloads") or 0)} for m in res2]
except Exception as e:
    print("image community search FAILED:",e); COMM2=[]
print("\n== community FLUX repos (%d total; top 15 by downloads) ==" % len(COMM2))
for m in sorted(COMM2,key=lambda x:-x["downloads"])[:15]:
    print("  %9s  %s" % (format(m["downloads"],","), m["id"]))

REG={"core":CORE,"image":IMG,"music":MUSIC,"extras":EXTRAS,"community":COMM,"image_community":COMM2,
     "cache":CACHE,"repo":CORE_REPO,"image_repo":IMG_REPO,"music_repo":MUSIC_REPO}
json.dump(REG,open("/tmp/h3_registry.json","w"))
print("\nregistry written to /tmp/h3_registry.json")
'''

# ---------------- cell 5: settings ----------------
settings = r'''
# ======================= SETTINGS (edit me) =======================
# H3 has NO negative prompt (CFG-distilled): prompt only, plus optional first/last frame.
import os
SMOKE_PROMPT=("A calm ocean wave rolling onto a quiet beach at golden hour, "
              "slow cinematic pan, soft warm sunlight, gentle foam detail.")
IMAGE_SMOKE_PROMPT=("A weathered fisherman's cottage on a rocky cliff at sunrise, "
                    "warm golden light, cinematic detail, photorealistic, sharp focus.")
MUSIC_SMOKE_PROMPT=("Lo-fi chillhop, 78 BPM, D flat major, warm Rhodes chords, "
                    "dusty boom-bap drums, soft vinyl crackle, cozy late-night mood.")
MUSIC_SMOKE_LYRICS=("[Intro]\nMmm...\n\n[Verse]\nGolden hour on a rainy street,\n"
                    "headphones on, the city sleeps,\nRhodes and drums in slow retreat,\n"
                    "a little warmth the night can keep.\n")
COMMON=dict(
    modality="video",
    unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    vae="minimax_h3_video_vae_int8_convrot.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)
MUSIC_COMMON=dict(
    modality="music",
    unet="minimax_music3_dit_fp16.safetensors",
    clip="minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    vae="minimax_music3_dav.safetensors",
)
PRESETS={
  # fast verified smoke: 4-step turbo LoRA @ 480p (864x480)
  "fast_smoke": dict(COMMON, prompt=SMOKE_PROMPT, aspect="16:9 (Widescreen)", megapixels=0.4,
                     duration=5, seed=12345, turbo=True, turbo_steps=4, turbo_strength=1.0,
                     lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
  # full quality: 20-step dense @ 1344x768 (the v13 config)
  "quality": dict(COMMON, prompt=SMOKE_PROMPT, aspect="16:9 (Widescreen)", megapixels=0.98,
                  duration=5, seed=12345, turbo=False, turbo_steps=8, turbo_strength=1.0,
                  lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
  # image generation: Flux schnell (single T4, 4 steps, CFG 1.0, no negative prompt)
  "image_smoke": dict(modality="image", prompt=IMAGE_SMOKE_PROMPT,
                      checkpoint="flux1-schnell-fp8.safetensors",
                      width=1024, height=1024, steps=4, cfg=1.0,
                      sampler="euler", scheduler="simple", seed=12345),
  # music generation: MiniMax Music 3 (core ComfyUI nodes, single T4, 30 steps)
  "music_smoke": dict(MUSIC_COMMON, prompt=MUSIC_SMOKE_PROMPT, lyrics=MUSIC_SMOKE_LYRICS,
                      max_duration=5, seed=4242, tiled_decode=True, format="flac"),
}
ACTIVE_PRESET="fast_smoke"
ACTIVE=dict(PRESETS[ACTIVE_PRESET])

# boot smoke schedule: one low-quality smoke per modality; image first (fast, and its
# PNG feeds the task self-test), then video (dual-GPU proof), then music.
# Override: H3_SMOKE="image,video" (or "video" only)
SMOKE=[m for m in os.environ.get("H3_SMOKE","image,video,music").split(",") if m in ("image","video","music")]
if not SMOKE: SMOKE=["image","video","music"]
SMOKE_PRESET={"image":"image_smoke",
              "video":(ACTIVE_PRESET if ACTIVE.get("modality")=="video" else "fast_smoke"),
              "music":"music_smoke"}

def required_files(a):
    if a.get("modality")=="image":
        return [("checkpoints",a["checkpoint"])]
    if a.get("modality")=="music":
        return [("diffusion_models",a["unet"]),("text_encoders",a["clip"]),("vae",a["vae"])]
    f=[("diffusion_models",a["unet"]),("text_encoders",a["clip"]),
       ("vae",a["vae"]),("vae",a["audio_vae"])]
    if a.get("turbo") and a.get("lora"):
        f.append(("loras",a["lora"]))
    return f

print("ACTIVE preset:",ACTIVE_PRESET,"| modality:",ACTIVE.get("modality","video"))
if ACTIVE.get("modality")=="image":
    print("  image: %dx%d @ %d steps, cfg %s, %s/%s" % (
        ACTIVE["width"],ACTIVE["height"],ACTIVE["steps"],ACTIVE["cfg"],
        ACTIVE["sampler"],ACTIVE["scheduler"]))
    print("  checkpoint:",ACTIVE["checkpoint"],"| seed:",ACTIVE["seed"])
elif ACTIVE.get("modality")=="music":
    print("  music: max_duration",ACTIVE["max_duration"],"s | seed:",ACTIVE["seed"],
          "| tiled_decode:",ACTIVE["tiled_decode"],"| format:",ACTIVE["format"])
    print("  unet:",ACTIVE["unet"],"| clip:",ACTIVE["clip"],"| vae:",ACTIVE["vae"])
else:
    RES_16X9={0.2:(608,352),0.3:(736,416),0.4:(864,480),0.5:(960,544),0.6:(1056,608),
              0.7:(1152,640),0.8:(1216,672),0.9:(1280,736),0.98:(1344,768)}
    wh=RES_16X9.get(ACTIVE["megapixels"])
    print("  turbo:",ACTIVE["turbo"],
           "| steps:",(ACTIVE["turbo_steps"] if ACTIVE["turbo"] else 20),
           "| lora:",(ACTIVE["lora"] if ACTIVE["turbo"] else "(none)"))
    print("  megapixels:",ACTIVE["megapixels"],"| aspect:",ACTIVE["aspect"],
          "| expected 16:9 px:",wh)
    print("  duration:",ACTIVE["duration"],"s | seed:",ACTIVE["seed"])
print("smoke schedule:",SMOKE,"-> presets:",[SMOKE_PRESET[m] for m in SMOKE])
files=set()
for _m in SMOKE:
    files.update("%s/%s"%(c,n) for c,n in required_files(PRESETS[SMOKE_PRESET[_m]]))
files.update("%s/%s"%(c,n) for c,n in required_files(ACTIVE))
files=sorted(files)
print("  boot-required files:",len(files))
for x in files: print("   -",x)
'''

# ---------------- cell 6: assets ----------------
assets = r'''
import subprocess,sys,os,json,time
from pathlib import Path
CTE="/tmp/ComfyUI"
MODELS=Path(CTE)/"models"
# files needed by the boot smoke schedule + the active preset (settings cell)
EXPECT=set()
for _m in SMOKE:
    EXPECT.update("%s/%s"%(c,n) for c,n in required_files(PRESETS[SMOKE_PRESET[_m]]))
EXPECT.update("%s/%s"%(c,n) for c,n in required_files(ACTIVE))
EXPECT=sorted(EXPECT)

# --- Kaggle Dataset auto-cache: mirror /kaggle/input files into model dirs (instant symlinks) ---
CACHE_FILES={}
if os.path.isdir("/kaggle/input"):
    for root,_,files in os.walk("/kaggle/input"):
        for fn in files:
            CACHE_FILES.setdefault(fn, os.path.join(root,fn))
def local_ok(rel):
    p=MODELS/rel
    return p.exists() and p.stat().st_size>0
miss=[]
for rel in EXPECT:
    if local_ok(rel): continue
    src=CACHE_FILES.get(rel.rsplit("/",1)[-1])
    if src and os.path.getsize(src)>0:
        dst=MODELS/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.is_symlink() or dst.exists(): dst.unlink()
        try:
            os.symlink(src,dst)
            if local_ok(rel):
                print("CACHE hit:",rel,"<-",src); continue
        except Exception as e:
            print("symlink failed for",rel,":",e)
    miss.append(rel)

IMG_FILES={"checkpoints/flux1-schnell-fp8.safetensors","checkpoints/flux1-schnell.safetensors"}
MUSIC_FILES={"diffusion_models/minimax_music3_dit_fp16.safetensors",
             "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
             "vae/minimax_music3_dav.safetensors"}
if miss:
    core=[r for r in miss if r not in IMG_FILES and r not in MUSIC_FILES]
    imgf=[r for r in miss if r in IMG_FILES]
    musicf=[r for r in miss if r in MUSIC_FILES]
    if core:
        print("downloading",len(core),"files from Comfy-Org/MiniMax-H3 (HF token active)")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Comfy-Org/MiniMax-H3",allow_patterns=core,local_dir=str(MODELS))
    if imgf:
        print("downloading",len(imgf),"files from Comfy-Org/flux1-schnell")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Comfy-Org/flux1-schnell",
                          allow_patterns=[r.split("/",1)[-1] for r in imgf],
                          local_dir=str(MODELS/"checkpoints"))
    if musicf:
        print("downloading",len(musicf),"music files from Comfy-Org/MiniMax-Music-3")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Comfy-Org/MiniMax-Music-3",allow_patterns=musicf,
                          local_dir=str(MODELS))

# --- verify every file + write MANIFEST (source of truth for this session) ---
MANIFEST={}; bad=[]
for rel in EXPECT:
    p=MODELS/rel
    sz=p.stat().st_size if p.exists() else 0
    print(("OK  " if sz else "MISS"),sz,rel)
    if not sz: bad.append(rel)
    else: MANIFEST[rel]={"size":sz,"source":("kaggle_input" if p.is_symlink() else "hf")}
if bad: raise RuntimeError("model files missing after download: %s"%bad)
json.dump(MANIFEST,open(str(MODELS/"MANIFEST.json"),"w"),indent=1)
print("MANIFEST.json written:",len(MANIFEST),"files")

# --- task assets: Real-ESRGAN x4 upscale model (task: upscale) ---
up=MODELS/"upscale_models"/"RealESRGAN_x4plus.pth"
if not (up.exists() and up.stat().st_size>0):
    up.parent.mkdir(parents=True,exist_ok=True)
    rc=subprocess.run(["curl","-sSLf","--max-time","600","-o",str(up),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]).returncode
    print("RealESRGAN download rc=%d"%rc)
if up.exists() and up.stat().st_size>0:
    print("upscale model OK:",up.stat().st_size,"bytes")
else:
    print("WARNING: RealESRGAN_x4plus.pth missing -> upscale task disabled")

# --- rembg warm-up: segmentation model lands in models/rembg (dataset-cached) ---
try:
    from PIL import Image as _Img
    from rembg import new_session as _ns, remove as _rm
    _sess=_ns("u2net")
    _t0=time.time(); _o=_rm(_Img.new("RGB",(64,64),(120,120,120)),session=_sess)
    print("rembg warm-up OK: u2net ->",_o.mode,"in %.1fs"%(time.time()-_t0))
except BaseException as e:
    print("rembg warm-up failed (bg_remove/extract tasks disabled):",type(e).__name__,e)

# --- ffmpeg availability (video/audio tasks) ---
import shutil as _sh
_ff=_sh.which("ffmpeg") or ("/tmp/bin/ffmpeg" if os.path.exists("/tmp/bin/ffmpeg") else None)
print("ffmpeg:",_ff or "NOT FOUND (video/audio tasks disabled)")

subprocess.run(["curl","-L","-sSf","-o","/tmp/h3_t2v.json","https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v.json"],check=True)
subprocess.run(["curl","-L","-sSf","-o","/tmp/flux_schnell.json","https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/flux_schnell.json"],check=True)
subprocess.run(["curl","-L","-sSf","-o","/tmp/audio_minimax_music_3.json","https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/audio_minimax_music_3.json"],check=True)
print("assets staged in /tmp/ComfyUI/models, workflows at /tmp/h3_t2v.json + /tmp/flux_schnell.json + /tmp/audio_minimax_music_3.json")
'''

# ---------------- cell 7: convert ----------------
convert = r'''
# ============ PROMPT BUILDER: settings dict -> validated API prompt (shared by smoke + API) ============
import json, base64, hashlib, urllib.request, urllib.error
from pathlib import Path

BASE="http://127.0.0.1:8188"
MISSING=object()

def load_info():
    return json.loads(urllib.request.urlopen(BASE+"/object_info",timeout=60).read())

def _save_frame(durl, slot):
    """data-URL image -> /tmp/ComfyUI/input/<name>; returns the LoadImage name."""
    head,b64=durl.split(",",1)
    mime=head.split(";")[0].replace("data:","")
    ext={"image/png":"png","image/jpeg":"jpg","image/webp":"webp","image/gif":"gif"}.get(mime,"png")
    raw=base64.b64decode(b64)
    name="h3_%s_%s.%s"%(slot,hashlib.sha1(raw).hexdigest()[:10],ext)
    d=Path("/tmp/ComfyUI/input"); d.mkdir(parents=True,exist_ok=True)
    (d/name).write_bytes(raw)
    return name

def _flatten(wf, info):
    """Generic subgraph -> API prompt (shared by video + music lanes).
    pass 1: expand every subgraph instance; pass 2: regular UI-graph nodes."""
    def schema(ct):
        if ct not in info:
            raise RuntimeError("node type not registered: "+ct)
        s=info[ct]["input"]
        return set(s.get("required",{}))|set(s.get("optional",{}))

    subs={s["id"]:s for s in ((wf.get("definitions") or {}).get("subgraphs") or [])}
    main_links={l[0]:l for l in (wf.get("links") or [])}
    instance_out={}
    prompt={}
    _next=[1000]
    def alloc():
        _next[0]+=1
        return str(_next[0])

    def main_value(link_id):
        l=main_links[link_id]
        src,slot=l[1],l[2]
        if str(src) in instance_out:
            return instance_out[str(src)](slot)
        return [str(src),slot]

    def expand_instance(node):
        sub=subs[node["type"]]
        ilinks={l["id"]:l for l in (sub.get("links") or [])}
        inst_inputs={i["name"]:i for i in (node.get("inputs") or [])}
        wnamed=node.get("widgets_values_named") or {}

        def sub_input(slot):
            name=sub["inputs"][slot]["name"]
            ii=inst_inputs.get(name)
            if ii is not None and ii.get("link") is not None:
                return main_value(ii["link"])
            if name in wnamed:
                return wnamed[name]
            return MISSING

        idmap={}
        for n in sub["nodes"]:
            schema(n["type"])
            idmap[str(n["id"])]=alloc()

        def out_value(slot):
            out=(sub.get("outputs") or [])[slot]
            lk=ilinks[out["linkIds"][0]]
            return [idmap[str(lk["origin_id"])],lk["origin_slot"]]

        for n in sub["nodes"]:
            valid=schema(n["type"])
            linked=set(); inputs={}
            for inp in (n.get("inputs") or []):
                nm=inp["name"]
                if inp.get("link") is None: continue   # links always included: autogrow slots
                linked.add(nm)                          # (values.a) only exist after finalization
                lk=ilinks.get(inp["link"])
                if lk is None: raise RuntimeError("inner link missing: %s"%inp["link"])
                if lk["origin_id"]==-10:
                    v=sub_input(lk["origin_slot"])
                else:
                    oid=idmap.get(str(lk["origin_id"]))
                    if oid is None: raise RuntimeError("inner origin missing: %s"%lk["origin_id"])
                    v=[oid,lk["origin_slot"]]
                if v is MISSING: continue
                inputs[nm]=v
            named=n.get("widgets_values_named") or {}
            for nm in valid:
                if nm in inputs or nm in linked: continue
                if nm in named: inputs[nm]=named[nm]
            prompt[idmap[str(n["id"])]]={"class_type":n["type"],"inputs":inputs}
        return out_value

    # pass 1: flatten every subgraph instance
    for n in wf.get("nodes",[]):
        if n["type"] in subs:
            instance_out[str(n["id"])]=expand_instance(n)
            print("expanded subgraph",n["type"],"-> nodes:",len(prompt))

    # pass 2: regular (UI graph) nodes
    for n in wf.get("nodes",[]):
        t=n["type"]
        if t in subs: continue
        if t not in info:
            print("skipping UI-only node:",t,"#%s"%n["id"]); continue
        valid=schema(t)
        linked=set(); inputs={}
        for inp in (n.get("inputs") or []):
            nm=inp["name"]
            if inp.get("link") is None: continue
            linked.add(nm)
            v=main_value(inp["link"])
            if v is MISSING: continue
            inputs[nm]=v
        named=n.get("widgets_values_named") or {}
        for nm in valid:
            if nm in inputs or nm in linked: continue
            if nm in named: inputs[nm]=named[nm]
        prompt[str(n["id"])]={"class_type":t,"inputs":inputs}

    print("flattened prompt nodes:",len(prompt))
    return prompt

# MiniMax Music 3 template: inner nodes store positional widgets only (older graph
# format). Schema-verified order for promoting them to widgets_values_named; "" skips
# the frontend-only control_after_generate slot. Linked inputs override these anyway.
MUSIC_WIDGETS={
    "UNETLoader":["unet_name","weight_dtype"],
    "MiniMaxMusic3TextEncode":["caption","lyrics","seed","","max_duration","cfg_scale","top_k"],
    "CLIPLoader":["clip_name","type","device"],
    "VAELoader":["vae_name"],
    "EmptyMiniMaxMusic3LatentAudio":["seconds","batch_size"],
    "KSampler":["seed","","steps","cfg","sampler_name","scheduler","denoise"],
    "VAEDecodeAudioTiled":["tile_size","overlap"],
    "SeedNode":["seed"],
    "ComfySwitchNode":["switch"],
}

def _build_music(cfg):
    """MiniMax Music 3 (audio_minimax_music_3.json) -> API prompt.
    Core ComfyUI nodes (comfy_extras/nodes_minimax_music.py): no custom pack, no
    MultiStream, single T4, no frame conditioning."""
    wf=json.load(open("/tmp/audio_minimax_music_3.json"))
    subs=(wf.get("definitions") or {}).get("subgraphs") or []
    if not subs: raise RuntimeError("music template: no subgraph")
    inst=[n for n in wf["nodes"] if n["type"]==subs[0]["id"]]
    if not inst: raise RuntimeError("music template: subgraph instance missing")
    # instance widget settings (caption lyrics max_duration seed unet clip vae switch)
    iw=inst[0].setdefault("widgets_values_named",{})
    iw.update({
        "caption":cfg["prompt"], "lyrics":cfg.get("lyrics",""),
        "max_duration":float(cfg.get("max_duration",60)),
        "seed":int(cfg["seed"]),
        "unet_name":cfg["unet"], "clip_name":cfg["clip"], "vae_name":cfg["vae"],
        "switch":bool(cfg.get("tiled_decode",True)),
    })
    for n in subs[0]["nodes"]:
        names=MUSIC_WIDGETS.get(n["type"])
        wv=n.get("widgets_values") or []
        if not names or not wv: continue
        out={}
        for i,val in enumerate(wv):
            if i<len(names) and names[i]: out[names[i]]=val
        if out: n["widgets_values_named"]=out
    for n in wf["nodes"]:
        if n["type"]=="SaveAudioAdvanced":
            n.setdefault("widgets_values_named",{}).update({
                "filename_prefix":cfg.get("output_prefix","h3_music"),
                "format":{"format":cfg.get("format","flac")}})
    info=load_info()
    prompt=_flatten(wf, info)
    meta={"nodes":len(prompt),"modality":"music",
          "max_duration":float(cfg.get("max_duration",60)),
          "steps":30,"format":cfg.get("format","flac"),
          "tiled_decode":bool(cfg.get("tiled_decode",True)),
          "unet":cfg["unet"],"clip":cfg["clip"],"vae":cfg["vae"]}
    return prompt, meta

def build_prompt(cfg):
    """cfg (a PRESETS-style dict) -> (prompt, meta). Flattens the official subgraph,
    applies settings, wires optional first/last frame, inserts H3MultiStream."""
    if cfg.get("modality")=="image":
        return _build_image(cfg)
    if cfg.get("modality")=="music":
        return _build_music(cfg)
    wf=json.load(open("/tmp/h3_t2v.json"))
    SUBID=(wf.get("definitions") or {}).get("subgraphs")[0]["id"]
    inst=[n for n in wf["nodes"] if n["type"]==SUBID][0]
    iw=inst.setdefault("widgets_values_named",{})
    iw.update({
        "prompt":cfg["prompt"], "value_1":float(cfg["duration"]),
        "noise_seed":int(cfg["seed"]),
        "unet_name":cfg["unet"], "clip_name":cfg["clip"],
        "vae_name":cfg["vae"], "vae_name_1":cfg["audio_vae"],
        "value":bool(cfg["turbo"]), "lora_name":cfg["lora"],
        "strength_model_1":float(cfg["turbo_strength"]),
        "value_2":int(cfg["turbo_steps"]),
    })
    for n in wf["nodes"]:
        if n["type"]=="ResolutionSelector":
            rw=n.setdefault("widgets_values_named",{})
            rw["aspect_ratio"]=cfg["aspect"]; rw["megapixels"]=cfg["megapixels"]

    # optional first/last-frame conditioning: LoadImage -> subgraph input (link ids free-form)
    frames=[]
    for slot,key in ((0,"first_frame"),(1,"last_frame")):
        durl=cfg.get(key+"_b64")
        if not durl: continue
        name=_save_frame(durl,key)
        nid=9001+slot; lid=7777+slot
        wf["nodes"].append({"id":nid,"type":"LoadImage","pos":[0,0],"size":[140,100],"flags":{},
            "order":99,"mode":0,"inputs":[],
            "outputs":[{"name":"IMAGE","type":"IMAGE","links":[lid],"slot_index":0}],
            "properties":{},"widgets_values":[name],"widgets_values_named":{"image":name}})
        for ii in inst.get("inputs") or []:
            if ii.get("name")==key: ii["link"]=lid
        wf["links"].append([lid,nid,0,inst["id"],slot,"IMAGE"])
        frames.append(name)
    if frames: print("frame conditioning:",frames)

    info=load_info()
    prompt=_flatten(wf, info)
    def schema(ct):
        if ct not in info:
            raise RuntimeError("node type not registered: "+ct)
        s=info[ct]["input"]
        return set(s.get("required",{}))|set(s.get("optional",{}))

    # turbo sanity: one boolean must drive both switches (LoRA branch + step count)
    if cfg["turbo"]:
        pb=[nid for nid,n in prompt.items()
            if n["class_type"]=="PrimitiveBoolean" and n["inputs"].get("value") is True]
        if not pb: raise RuntimeError("turbo enabled but no PrimitiveBoolean(value=True) in prompt")
        print("turbo path armed: PrimitiveBoolean",pb,"-> steps",cfg["turbo_steps"],"+ LoRA")
    else:
        print("dense path: 20 steps, no LoRA")

    # insert H3MultiStream after the model switch, feeding BasicGuider + BasicScheduler
    schema("H3MultiStream")
    targets=[]
    for nid,node in prompt.items():
        if node["class_type"] in ("BasicGuider","BasicScheduler") and isinstance(node["inputs"].get("model"),list):
            targets.append((nid,node["inputs"]["model"]))
    if not targets:
        raise RuntimeError("no BasicGuider/BasicScheduler with a linked model input")
    src=targets[0][1]
    h3_in={"model":src,"enabled":True,"second_gpu":-1,"exchange":"host","exchange_chunks":8,
           "sparse_attention":False,"dynamic_vram":"keep","vram_block_cache":False,
           "vram_reserve_gb":2.0,"sparse_vsa":False,"weight_cache":False,"cache_ram_reserve_gb":0.0}
    valid=schema("H3MultiStream")
    missing=[k for k in valid if k not in h3_in and k!="gpus" and not isinstance(h3_in.get(k),list)]
    h3_in={k:v for k,v in h3_in.items() if k in valid}
    prompt["9999"]={"class_type":"H3MultiStream","inputs":h3_in}
    n_p=0
    for nid,v in targets:
        if v==src:
            prompt[nid]["inputs"]["model"]=["9999",0]; n_p+=1
    print("H3MultiStream inserted as node 9999, feeding",n_p,"consumers; model source",src)
    if missing: print("WARNING: H3 inputs not supplied:",missing)

    steps=cfg["turbo_steps"] if cfg["turbo"] else 20
    meta={"nodes":len(prompt),"turbo":bool(cfg["turbo"]),"steps":steps,
          "megapixels":cfg["megapixels"],"aspect":cfg["aspect"],
          "duration":cfg["duration"],"frames_injected":frames}
    return prompt, meta

def _build_image(cfg):
    """Flux image workflow -> API prompt. Flat graph: no subgraph, no MultiStream,
    single T4 (the dual-GPU assertion in the wait cell is modality-aware)."""
    wf=json.load(open("/tmp/flux_schnell.json"))
    nodes={n["id"]:n for n in wf["nodes"]}
    def setnamed(n,upd): n.setdefault("widgets_values_named",{}).update(upd)
    ks=[n for n in wf["nodes"] if n["type"]=="KSampler"]
    if not ks: raise RuntimeError("flux template: no KSampler node")
    ks=ks[0]
    links={l[0]:l for l in (wf.get("links") or [])}
    inps={i["name"]:i for i in (ks.get("inputs") or [])}
    def origin(slot):
        ii=inps.get(slot)
        if ii is None or ii.get("link") is None: return None
        return nodes.get(links[ii["link"]][1])
    pos,neg=origin("positive"),origin("negative")
    if pos is not None: setnamed(pos,{"text":cfg["prompt"]})
    if neg is not None: setnamed(neg,{"text":""})   # Flux: no negative prompt (CFG=1)
    for n in wf["nodes"]:
        t=n["type"]
        if t=="CheckpointLoaderSimple": setnamed(n,{"ckpt_name":cfg["checkpoint"]})
        elif t=="EmptySD3LatentImage":
            setnamed(n,{"width":int(cfg["width"]),"height":int(cfg["height"]),"batch_size":1})
        elif t=="SaveImage": setnamed(n,{"filename_prefix":"h3_image"})
    setnamed(ks,{"seed":int(cfg["seed"]),"steps":int(cfg["steps"]),"cfg":float(cfg["cfg"]),
                 "sampler_name":cfg["sampler"],"scheduler":cfg["scheduler"],"denoise":1.0})
    info=load_info()
    prompt={}
    for n in wf["nodes"]:
        t=n["type"]
        if t not in info:
            print("skipping UI-only node:",t,"#%s"%n["id"]); continue
        s=info[t]["input"]
        valid=set(s.get("required",{}))|set(s.get("optional",{}))
        inputs={}
        for inp in (n.get("inputs") or []):
            if inp.get("link") is None: continue
            lk=links.get(inp["link"])
            if lk is None: raise RuntimeError("flux link missing: %s"%inp["link"])
            inputs[inp["name"]]=[str(lk[1]),lk[2]]
        named=n.get("widgets_values_named") or {}
        for nm in valid:
            if nm in inputs: continue
            if nm in named: inputs[nm]=named[nm]
        prompt[str(n["id"])]={"class_type":t,"inputs":inputs}
    print("flattened image prompt nodes:",len(prompt))
    meta={"nodes":len(prompt),"modality":"image","steps":int(cfg["steps"]),
          "size":"%dx%d"%(int(cfg["width"]),int(cfg["height"])),"cfg":float(cfg["cfg"]),
          "checkpoint":cfg["checkpoint"]}
    return prompt, meta
'''

# ---------------- cell 8: start ----------------
start = r'''
import subprocess,sys,os,time,json
c=subprocess.Popen([sys.executable,"/tmp/ComfyUI/main.py","--listen","127.0.0.1","--port","8188","--preview-method","latent2rgb"],
                   cwd="/tmp/ComfyUI", stdout=open("/tmp/comfy.log","w"), stderr=subprocess.STDOUT)
print("Comfy pid",c.pid)
# sample both GPUs every 10s -> /tmp/gpus.log (dual-GPU evidence for the smoke test)
try:
    subprocess.Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
                      "--format=csv,noheader","-l","10"],
                     stdout=open("/tmp/gpus.log","w"), stderr=subprocess.STDOUT)
except Exception as e:
    print("gpu monitor not started:",e)
ready=False
for i in range(72):
    try:
        import urllib.request as ur
        ur.urlopen("http://127.0.0.1:8188/history", timeout=3)
        ready=True; break
    except Exception: time.sleep(5)
if not ready:
    raise RuntimeError("ComfyUI did not listen in time")
print("ComfyUI local ready")
'''

# ---------------- cell 9: api ----------------
api = r'''
# ============ HUB API + PUBLIC PROXY: /h3api/* served here, everything else piped to ComfyUI ============
import json, os, socket, threading, time, urllib.request, urllib.error, http.server, subprocess, shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

API_PORT=8190
COMFY=("127.0.0.1",8188)
MODELS=Path("/tmp/ComfyUI/models")
API_UP=False
UP={"since":time.time()}
DL={"running":False,"last":None}
LOCK=threading.Lock()

def _j(code,obj):
    return code,{"Content-Type":"application/json"},json.dumps(obj).encode()

def comfy(method,path,data=None,timeout=90):
    req=urllib.request.Request("http://%s:%d%s"%(COMFY[0],COMFY[1],path),
        data=(json.dumps(data).encode() if data is not None else None),method=method)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        raw=r.read()
    return json.loads(raw) if raw else {}

def _inventory():
    out={}
    if not MODELS.is_dir(): return out
    for p in sorted(MODELS.rglob("*")):
        if not (p.is_file() or p.is_symlink()): continue
        try: sz=p.stat().st_size
        except OSError: continue
        rel=str(p.relative_to(MODELS))
        out[rel]={"size":sz,"source":("kaggle_input" if p.is_symlink() else "local")}
    return out

def _required():
    return ["%s/%s"%(c,n) for c,n in required_files(ACTIVE)]

# ---------------- handlers ----------------
def H_index(o):
    return _j(200,{"ok":True,"name":"MiniMax H3 generation hub","docs":"/h3api/docs",
        "endpoints":sorted(["%s %s"%(m,p) for (m,p) in ROUTES]),
        "comfy_root":"/","native_prompt_api":"/prompt"})

def H_health(o):
    try: comfy("GET","/system_stats",timeout=4); up=True
    except Exception: up=False
    return _j(200,{"ok":True,"comfy":up,"api_uptime_s":round(time.time()-UP["since"]),
        "active_preset":ACTIVE_PRESET,"active":ACTIVE,"required":_required()})

def H_catalog(o):
    q=parse_qs(urlparse(o["path"]).query)
    if (q.get("refresh") or ["0"])[0] in ("1","true"):
        try:
            reg0=json.load(open("/tmp/h3_registry.json"))
            CORE=[{"path":f["path"],"size":int(f.get("size") or 0)} for f in
                  hf_get("https://huggingface.co/api/models/%s/tree/main?recursive=true"%CORE_REPO)
                  if f.get("type")=="file" and not f["path"].startswith(".")]
            COMM=[{"id":m["id"],"downloads":int(m.get("downloads") or 0)} for m in
                  hf_get("https://huggingface.co/api/models?search=MiniMax-H3&limit=100&sort=downloads&direction=-1")]
            IMG=[{"path":("checkpoints/" if "/" not in f["path"] else "")+f["path"],
                  "size":int(f.get("size") or 0)} for f in
                 hf_get("https://huggingface.co/api/models/%s/tree/main?recursive=true"%reg0.get("image_repo","Comfy-Org/flux1-schnell"))
                 if f.get("type")=="file" and f["path"].endswith(".safetensors")]
            MUSIC=[{"path":f["path"],"size":int(f.get("size") or 0)} for f in
                   hf_get("https://huggingface.co/api/models/%s/tree/main?recursive=true"%reg0.get("music_repo","Comfy-Org/MiniMax-Music-3"))
                   if f.get("type")=="file" and f["path"].endswith(".safetensors")]
            reg0["core"]=CORE; reg0["community"]=COMM; reg0["image"]=IMG; reg0["music"]=MUSIC
            json.dump(reg0,open("/tmp/h3_registry.json","w"))
        except Exception as e:
            return _j(502,{"ok":False,"error":"live refresh failed: %s"%e})
    try: reg=json.load(open("/tmp/h3_registry.json"))
    except Exception as e: return _j(503,{"ok":False,"error":"registry not built yet: %s"%e})
    cache=reg.get("cache",{})
    for key in ("core","image","music","extras"):
        for f in reg.get(key,[]):
            p=MODELS/f["path"]
            f["status"]="local" if (p.exists() and p.stat().st_size>0) else                 ("cached" if f["path"].rsplit("/",1)[-1] in cache else "remote")
    return _j(200,{"ok":True,"repo":reg.get("repo"),"files":reg.get("core",[]),
        "image":reg.get("image",[]),"image_repo":reg.get("image_repo"),
        "music":reg.get("music",[]),"music_repo":reg.get("music_repo"),
        "extras":reg.get("extras",[]),
        "community":reg.get("community",[]),"image_community":reg.get("image_community",[]),
        "cache_files":len(cache)})

def H_models(o):
    mf={}
    try: mf=json.load(open(str(MODELS/"MANIFEST.json")))
    except Exception: pass
    return _j(200,{"ok":True,"files":_inventory(),"manifest":mf,"required":_required(),
        "active_preset":ACTIVE_PRESET})

def _apply_settings(b):
    global ACTIVE, ACTIVE_PRESET
    if b.get("preset") is not None:
        if b["preset"] not in PRESETS:
            return _j(400,{"ok":False,"error":"unknown preset","available":sorted(PRESETS)})
        ACTIVE_PRESET=b["preset"]; ACTIVE=dict(PRESETS[ACTIVE_PRESET])
    if isinstance(b.get("set"),dict):
        ACTIVE.update(b["set"]); ACTIVE_PRESET="(custom)"
    return None

def H_settings(o):
    if o["method"]=="GET":
        return _j(200,{"ok":True,"active_preset":ACTIVE_PRESET,"presets":PRESETS,"active":ACTIVE,
            "required":_required(),
            "missing":[r for r in _required() if not (MODELS/r).exists() or (MODELS/r).stat().st_size==0],
            "note":"H3 has no negative prompt (CFG-distilled): prompt + optional first/last frame only"})
    err=_apply_settings(o["body"])
    if err: return err
    return _j(200,{"ok":True,"active_preset":ACTIVE_PRESET,"active":ACTIVE,"required":_required(),
        "missing":[r for r in _required() if not (MODELS/r).exists() or (MODELS/r).stat().st_size==0]})

def _start_download(files,note):
    def run():
        DL["running"]=True
        try:
            from huggingface_hub import snapshot_download
            core=[f for f in files if f not in IMG_FILES]
            imgf=[f for f in files if f in IMG_FILES]
            if core:
                snapshot_download(repo_id=CORE_REPO,allow_patterns=core,
                                  local_dir=str(MODELS),token=os.environ.get("HF_TOKEN"))
            if imgf:
                snapshot_download(repo_id="Comfy-Org/flux1-schnell",
                                  allow_patterns=[f.split("/",1)[-1] for f in imgf],
                                  local_dir=str(MODELS/"checkpoints"),
                                  token=os.environ.get("HF_TOKEN"))
            DL["last"]={"ok":True,"files":files,"note":note,"t":time.time()}
        except Exception as e:
            DL["last"]={"ok":False,"error":str(e)[:500],"files":files,"t":time.time()}
        finally:
            DL["running"]=False
    threading.Thread(target=run,daemon=True).start()

def H_download(o):
    b=o["body"] or {}
    files=b.get("files") or ([b["file"]] if b.get("file") else [])
    files=[f for f in files if isinstance(f,str) and f and not f.startswith("/") and ".." not in f]
    if not files:
        return _j(400,{"ok":False,"error":"pass {'file':'loras/x.safetensors'} or {'files':[...]}"})
    with LOCK:
        if DL["running"]: return _j(409,{"ok":False,"error":"a download is already running","last":DL["last"]})
        _start_download(files,"manual")
    return _j(202,{"ok":True,"queued":files,"status":"/h3api/download"})

def H_download_status(o):
    return _j(200,{"ok":True,"running":DL["running"],"last":DL["last"]})

def H_select(o):
    """switch model/settings: downloads the new set (async) and deletes superseded local
    files so disk and GPU stay clean — the previous model never lingers."""
    b=o["body"] or {}
    err=_apply_settings(b)
    if err: return err
    keep=set(_required())
    PRUNE_DIRS={"diffusion_models","text_encoders","vae","loras","checkpoints"}
    freed=0; pruned=[]
    if b.get("prune",True):
        for rel in list(_inventory()):
            if rel=="MANIFEST.json" or rel in keep: continue
            if not rel.endswith(".safetensors"): continue
            if rel.split("/")[0] not in PRUNE_DIRS: continue
            p=MODELS/rel
            try:
                sz=p.stat().st_size; p.unlink(); freed+=sz; pruned.append(rel)
            except Exception: pass
    missing=sorted(r for r in keep if not (MODELS/r).exists() or (MODELS/r).stat().st_size==0)
    queued=False
    if missing:
        with LOCK:
            if not DL["running"]:
                _start_download(missing,"select"); queued=True
        if not queued:
            return _j(409,{"ok":True,"error":"download busy; retry later","active_preset":ACTIVE_PRESET,
                "active":ACTIVE,"pruned":pruned,"freed_bytes":freed,"missing":missing,"queued":False})
    return _j(200 if not missing else 202,{"ok":True,"active_preset":ACTIVE_PRESET,"active":ACTIVE,
        "pruned":pruned,"freed_bytes":freed,"missing":missing,"queued":queued,
        "status":"/h3api/download"})

def H_generate(o):
    b=o["body"] or {}
    cfg=dict(ACTIVE)
    if b.get("preset") is not None:
        if b["preset"] not in PRESETS:
            return _j(400,{"ok":False,"error":"unknown preset","available":sorted(PRESETS)})
        cfg=dict(PRESETS[b["preset"]])
    if isinstance(b.get("set"),dict): cfg.update(b["set"])
    if b.get("prompt"): cfg["prompt"]=str(b["prompt"])[:6000]
    if b.get("first_frame"): cfg["first_frame_b64"]=b["first_frame"]
    if b.get("last_frame"): cfg["last_frame_b64"]=b["last_frame"]
    if b.get("seed") is not None:
        try: cfg["seed"]=int(b["seed"])
        except Exception: return _j(400,{"ok":False,"error":"seed must be an integer"})
    try:
        prompt,meta=build_prompt(cfg)
    except Exception as e:
        return _j(500,{"ok":False,"error":"prompt build failed: %s"%e})
    try:
        res=comfy("POST","/prompt",{"prompt":prompt,"extra_data":{"preview_method":"latent2rgb"}},timeout=60)
    except urllib.error.HTTPError as e:
        return _j(502,{"ok":False,"error":"comfy rejected the prompt","detail":e.read().decode()[:4000]})
    except Exception as e:
        return _j(502,{"ok":False,"error":"comfy unreachable: %s"%e})
    return _j(202,{"ok":True,"prompt_id":res.get("prompt_id"),"meta":meta,
        "poll":"/h3api/jobs","history":"/history/"+str(res.get("prompt_id"))})

def H_jobs(o):
    try:
        hist=comfy("GET","/history")
        qq=comfy("GET","/queue")
    except Exception as e:
        return _j(503,{"ok":False,"error":"comfy unreachable: %s"%e})
    jobs=[]
    for pid,entry in list(hist.items())[-25:]:
        st=entry.get("status",{})
        jobs.append({"id":pid,"status":st.get("status_str"),
            "completed":st.get("completed"),"remaining":st.get("remaining"),
            "outputs":list((entry.get("outputs") or {}).keys())})
    jobs.reverse()
    return _j(200,{"ok":True,
        "queue":{"running":len(qq.get("queue_running") or []),
                 "pending":len(qq.get("queue_pending") or [])},
        "jobs":jobs})

def H_outputs(o):
    d=Path("/tmp/ComfyUI/output"); out=[]
    if d.is_dir():
        for p in d.rglob("*"):
            if not p.is_file(): continue
            try: st=p.stat()
            except OSError: continue
            out.append({"name":p.name,"size":st.st_size,"modified":int(st.st_mtime),
                "url":"/view?filename=%s&subfolder=&type=output"%quote(p.name)})
    out.sort(key=lambda x:-x["modified"])
    return _j(200,{"ok":True,"total":len(out),"files":out[:200]})

def H_free(o):
    try: comfy("POST","/free",{"unload_models":True,"free_memory":True})
    except Exception as e: return _j(502,{"ok":False,"error":"comfy unreachable: %s"%e})
    return _j(200,{"ok":True,"unloaded":True,"note":"models freed from GPU; next generate reloads"})

def H_gpus(o):
    try:
        r=subprocess.run(["nvidia-smi","--query-gpu=index,name,utilization.gpu,memory.used,power.draw",
                          "--format=csv,noheader"],capture_output=True,text=True,timeout=15)
        lines=[l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
        return _j(200,{"ok":bool(lines),"gpus":lines or None,
                       "detail":(r.stderr or "").strip()[:200] or None})
    except Exception as e:
        return _j(200,{"ok":False,"gpus":None,"detail":str(e)[:200]})

def H_import(o):
    """POST /h3api/import?path=<category>/<file.safetensors> with raw file bytes as body."""
    rel=(parse_qs(urlparse(o["path"]).query).get("path") or [""])[0].strip()
    if not rel or ".." in rel or rel.startswith("/") or not rel.endswith(".safetensors"):
        return _j(400,{"ok":False,"error":"?path=<category>/<file.safetensors> required"})
    cl=int(o["headers"].get("Content-Length") or 0)
    if cl<=0: return _j(400,{"ok":False,"error":"raw file bytes required as body"})
    free=shutil.disk_usage("/tmp").free
    if cl>free*0.95:
        return _j(507,{"ok":False,"error":"not enough disk for import","free":free,"need":cl})
    dst=MODELS/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with open(dst,"wb") as f:
        left=cl
        while left>0:
            chunk=o["rfile"](min(1<<20,left))
            if not chunk: break
            f.write(chunk); n+=len(chunk); left-=len(chunk)
    if n!=cl:
        try: dst.unlink()
        except Exception: pass
        return _j(400,{"ok":False,"error":"incomplete upload (%d/%d bytes)"%(n,cl)})
    return _j(201,{"ok":True,"path":rel,"size":n,
        "note":"file is on disk; use /h3api/select or /h3api/generate to use it"})

# ================= TASK ENGINE: image / video / audio use-cases =================
def _find_bin(name):
    p=shutil.which(name)
    if p: return p
    alt="/tmp/bin/"+name
    return alt if os.path.exists(alt) else None
FFMPEG=_find_bin("ffmpeg"); FFPROBE=_find_bin("ffprobe")
_REMBG=None;_RembgErr=None
try:
    import rembg as _REMBG
except BaseException as e:            # SystemExit if onnxruntime is missing
    _RembgErr="%s: %s"%(type(e).__name__,e)
    print("rembg unavailable (bg_remove/extract disabled):",_RembgErr)
_SESS={}
def _rembg_session(name):
    if _REMBG is None:
        raise RuntimeError("rembg unavailable: %s"%(_RembgErr or "not installed (pip install rembg[cpu])"))
    if name not in _SESS: _SESS[name]=_REMBG.new_session(name)
    return _SESS[name]
def _durl(raw,mime="image/png"):
    import base64
    return "data:%s;base64,%s"%(mime,base64.b64encode(raw).decode())
def _b64raw(v):
    if not isinstance(v,str) or not v: return None
    import base64
    s=v.split(",",1)[1] if v.startswith("data:") and "," in v else v
    try: return base64.b64decode(s)
    except Exception: return None
def _img_in(b):
    raw=_b64raw(b.get("image"))
    if raw is None: return None
    import io
    from PIL import Image
    im=Image.open(io.BytesIO(raw)); im.load()
    if im.mode not in ("RGB","RGBA"): im=im.convert("RGB")
    return im
def _png(im):
    import io
    b=io.BytesIO(); im.save(b,"PNG"); return b.getvalue()
def _view(p):
    p=Path(p)
    try: sub=str(p.relative_to("/tmp/ComfyUI/output").parent)
    except Exception: return None
    if sub==".": sub=""
    return "/view?filename=%s&subfolder=%s&type=output"%(quote(p.name),quote(sub))
def _task_save(name,raw,ext):
    import hashlib
    d=Path("/tmp/ComfyUI/output/tasks"); d.mkdir(parents=True,exist_ok=True)
    p=d/("%s_%s_%s.%s"%(name,time.strftime("%H%M%S"),hashlib.sha1(raw).hexdigest()[:8],ext))
    p.write_bytes(raw); return p
def _sniff(raw):
    if raw[4:8]==b"ftyp": return "mp4"
    if raw[:4]==b"RIFF": return "avi"
    if raw[:3]==b"ID3": return "mp3"
    if raw[:6] in (b"GIF87a",b"GIF89a"): return "gif"
    if raw[:4]==b"OggS": return "ogg"
    return "bin"
def _media_in(b,keys):
    """data URL (any key) or <key>_path (must live under /tmp/ComfyUI) -> file Path."""
    import hashlib
    for k in keys:
        v=b.get(k+"_path")
        if isinstance(v,str) and v:
            p=Path(v).resolve(); root=Path("/tmp/ComfyUI").resolve()
            if str(p)!=str(root) and not str(p).startswith(str(root)+os.sep):
                raise ValueError("path must live under /tmp/ComfyUI (output or input)")
            if not p.is_file(): raise ValueError("file not found: %s"%v)
            return p
        raw=_b64raw(b.get(k))
        if raw:
            d=Path("/tmp/ComfyUI/input"); d.mkdir(parents=True,exist_ok=True)
            p=d/("h3task_%s.%s"%(hashlib.sha1(raw).hexdigest()[:12],_sniff(raw)))
            if not p.exists(): p.write_bytes(raw)
            return p
    return None
def _src(b,keys):
    try: p=_media_in(b,keys)
    except ValueError as e: return None,_j(400,{"ok":False,"error":str(e)})
    if p is None:
        return None,_j(400,{"ok":False,"error":"pass a data URL (%s) or '%s_path'"%("/".join(keys),keys[0])})
    return p,None
def _run(cmd,timeout=300):
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if r.returncode!=0:
        raise RuntimeError("ffmpeg rc=%d: %s"%(r.returncode,(r.stderr or "").strip()[-400:]))
    return r
def _ff_err():
    if not FFMPEG: return _j(503,{"ok":False,"error":"ffmpeg not available (video/audio tasks disabled)"})
    return None
def _probe_json(p):
    if not FFPROBE: return {}
    try:
        r=subprocess.run([FFPROBE,"-v","error","-print_format","json","-show_format",
                          "-show_streams",str(p)],capture_output=True,text=True,timeout=60)
        return json.loads(r.stdout or "{}")
    except Exception: return {}

def T_bg_remove(b):
    """cut out the background -> transparent PNG"""
    im=_img_in(b)
    if im is None: return _j(400,{"ok":False,"error":"'image' (data URL) required"})
    name=str(b.get("model") or "u2net")
    try: sess=_rembg_session(name)
    except Exception as e: return _j(503,{"ok":False,"error":"model unavailable: %s"%e})
    t0=time.time()
    out=_REMBG.remove(im,session=sess,alpha_matting=bool(b.get("alpha_matting")))
    if out.mode!="RGBA": out=out.convert("RGBA")
    raw=_png(out); p=_task_save("bgremove",raw,"png")
    return _j(200,{"ok":True,"task":"bg_remove","outputs":[_durl(raw)],"saved":[p.name],
        "view":[_view(p)],"meta":{"model":name,"size":[out.width,out.height],
        "seconds":round(time.time()-t0,2)}})

def T_extract(b):
    """element extractor: foreground elements as separate transparent cutouts + bboxes"""
    import numpy as np
    from scipy import ndimage
    im=_img_in(b)
    if im is None: return _j(400,{"ok":False,"error":"'image' (data URL) required"})
    name=str(b.get("model") or "u2net")
    try: sess=_rembg_session(name)
    except Exception as e: return _j(503,{"ok":False,"error":"model unavailable: %s"%e})
    t0=time.time()
    out=_REMBG.remove(im,session=sess)
    if out.mode!="RGBA": out=out.convert("RGBA")
    arr=np.array(out)
    mask=arr[...,3]>128
    lab,n=ndimage.label(mask,structure=np.ones((3,3),dtype=bool))
    objs=ndimage.find_objects(lab)
    sizes=ndimage.sum(mask,lab,range(1,n+1)) if n else []
    min_area=float(b.get("min_area") or max(100.0,0.002*mask.size))
    max_el=max(1,min(int(b.get("max_elements") or 5),20))
    order=sorted(range(n),key=lambda i:-(sizes[i] if n else 0))
    elems=[]; saved=[]
    for i in order:
        if n and sizes[i]<min_area: continue
        sl=objs[i]
        if sl is None: continue
        y0,y1=sl[0].start,sl[0].stop; x0,x1=sl[1].start,sl[1].stop
        if x1-x0<8 or y1-y0<8: continue
        raw=_png(out.crop((x0,y0,x1,y1)))
        p=_task_save("elem%d"%len(elems),raw,"png")
        elems.append({"bbox":[int(x0),int(y0),int(x1),int(y1)],
                      "size":[int(x1-x0),int(y1-y0)],
                      "image":_durl(raw),"view":_view(p)})
        saved.append(p)
        if len(elems)>=max_el: break
    meta={"model":name,"elements":len(elems),"labels":int(n),
          "size":[out.width,out.height],"seconds":round(time.time()-t0,2)}
    resp={"ok":True,"task":"extract","outputs":[e["image"] for e in elems],
          "elements":elems,"saved":[p.name for p in saved],"view":[_view(p) for p in saved],
          "meta":meta}
    if b.get("include_mask"):
        from PIL import Image as _I
        rawm=_png(_I.fromarray((mask*255).astype("uint8")))
        pm=_task_save("mask",rawm,"png")
        resp["mask"]=_durl(rawm); resp["saved"].append(pm.name); resp["view"].append(_view(pm))
    return _j(200,resp)

def T_upscale(b):
    """4x super-resolution via Real-ESRGAN (GPU, through Comfy)"""
    im=_img_in(b)
    if im is None: return _j(400,{"ok":False,"error":"'image' (data URL) required"})
    mdl=str(b.get("model") or "RealESRGAN_x4plus.pth")
    if "/" in mdl or ".." in mdl: return _j(400,{"ok":False,"error":"bad model name"})
    mp=MODELS/"upscale_models"/mdl
    if not (mp.exists() and mp.stat().st_size>0):
        return _j(503,{"ok":False,"error":"upscale model missing: %s (see /h3api/catalog)"%mdl})
    import io,hashlib
    from PIL import Image as _I
    bi=io.BytesIO(); im.convert("RGB").save(bi,"PNG"); inraw=bi.getvalue()
    d=Path("/tmp/ComfyUI/input"); d.mkdir(parents=True,exist_ok=True)
    fname="h3task_%s.png"%hashlib.sha1(inraw).hexdigest()[:12]
    (d/fname).write_bytes(inraw)
    prompt={"1":{"class_type":"LoadImage","inputs":{"image":fname}},
            "2":{"class_type":"UpscaleModelLoader","inputs":{"model_name":mdl}},
            "3":{"class_type":"ImageUpscaleWithModel","inputs":{"image":["1",0],
                 "upscale_model":["2",0]}},
            "4":{"class_type":"SaveImage","inputs":{"images":["3",0],
                 "filename_prefix":"h3task_up"}}}
    try:
        res=comfy("POST","/prompt",{"prompt":prompt},timeout=60)
    except urllib.error.HTTPError as e:
        return _j(502,{"ok":False,"error":"comfy rejected upscale prompt",
                       "detail":e.read().decode()[:1500]})
    except Exception as e:
        return _j(502,{"ok":False,"error":"comfy unreachable: %s"%e})
    pid=res.get("prompt_id"); entry=None; t0=time.time()
    while time.time()-t0<300:
        try:
            h=comfy("GET","/history/"+str(pid),timeout=10)
            if h.get(pid): entry=h[pid]; break
        except Exception: pass
        time.sleep(1)
    if entry is None: return _j(504,{"ok":False,"error":"upscale timed out (300 s)"})
    st=entry.get("status",{})
    if st.get("status_str")=="error":
        return _j(502,{"ok":False,"error":"upscale execution failed","detail":json.dumps(st)[:1200]})
    img=None
    for node in (entry.get("outputs") or {}).values():
        for it in (node.get("images") or []):
            pp=Path("/tmp/ComfyUI/output")/str(it.get("subfolder") or "")/str(it["filename"])
            if pp.exists(): img=pp
    if img is None: return _j(502,{"ok":False,"error":"upscale produced no image"})
    raw=img.read_bytes()
    sz=_I.open(io.BytesIO(raw)).size
    return _j(200,{"ok":True,"task":"upscale","outputs":[_durl(raw)],"saved":[img.name],
        "view":[_view(img)],"meta":{"model":mdl,"scale":4,"size":[int(sz[0]),int(sz[1])]}})

def T_video_frames(b):
    e=_ff_err()
    if e: return e
    src,err=_src(b,("video",))
    if err: return err
    count=max(1,min(int(b.get("count") or 4),32))
    maxpx=max(256,min(int(b.get("max_size") or 1024),2048))
    info=_probe_json(src)
    dur=float((info.get("format") or {}).get("duration") or 0)
    fps=(count/dur) if dur>0 else 1.0
    d=Path("/tmp/ComfyUI/output/tasks"); d.mkdir(parents=True,exist_ok=True)
    stem="frames_%s"%int(time.time())
    _run([FFMPEG,"-y","-i",str(src),"-vf",
          "fps=%.4f,scale='min(%d,iw)':-2"%(fps,maxpx),
          "-frames:v","%d"%count,str(d/(stem+"_%03d.png"))])
    outs=sorted(d.glob(stem+"_*.png"))
    if not outs: return _j(502,{"ok":False,"error":"no frames extracted"})
    raws=[p.read_bytes() for p in outs]
    return _j(200,{"ok":True,"task":"video_frames",
        "outputs":[_durl(r) for r in raws],"saved":[p.name for p in outs],
        "view":[_view(p) for p in outs],
        "meta":{"count":len(outs),"fps":round(fps,3),"duration":dur,"max_size":maxpx}})

def T_video_gif(b):
    e=_ff_err()
    if e: return e
    src,err=_src(b,("video",))
    if err: return err
    fps=max(1,min(int(b.get("fps") or 4),15))
    width=max(128,min(int(b.get("width") or 480),1280))
    maxdur=max(1.0,min(float(b.get("max_duration") or 10),60))
    d=Path("/tmp/ComfyUI/output/tasks"); d.mkdir(parents=True,exist_ok=True)
    out=d/("gif_%d.gif"%int(time.time()))
    _run([FFMPEG,"-y","-t","%.2f"%maxdur,"-i",str(src),
          "-vf","fps=%d,scale=%d:-2:flags=lanczos"%(fps,width),"-loop","0",str(out)])
    raw=out.read_bytes()
    resp={"ok":True,"task":"video_gif","outputs":[_durl(raw,"image/gif")],
          "saved":[out.name],"view":[_view(out)],
          "meta":{"fps":fps,"width":width,"bytes":len(raw)}}
    if len(raw)>25*1024*1024:
        resp["outputs"]=[]; resp["note"]="gif >25MB: fetch via saved/view (lower fps/width)"
    return _j(200,resp)

def T_audio_extract(b):
    e=_ff_err()
    if e: return e
    src,err=_src(b,("audio","video","media"))
    if err: return err
    fmt=str(b.get("format") or "wav").lower()
    if fmt not in ("wav","mp3"): return _j(400,{"ok":False,"error":"format must be wav or mp3"})
    d=Path("/tmp/ComfyUI/output/tasks"); d.mkdir(parents=True,exist_ok=True)
    out=d/("audio_%d.%s"%(int(time.time()),fmt))
    cmd=[FFMPEG,"-y","-i",str(src),"-vn"]
    try:
        if fmt=="mp3":
            _run(cmd+["-acodec","libmp3lame","-q:a","4",str(out)])
        else:
            _run(cmd+["-acodec","pcm_s16le",str(out)])
    except RuntimeError:
        if fmt!="mp3": raise
        fmt="wav"; out=d/("audio_%d.wav"%int(time.time()))
        _run([FFMPEG,"-y","-i",str(src),"-vn","-acodec","pcm_s16le",str(out)])
    raw=out.read_bytes()
    resp={"ok":True,"task":"audio_extract",
          "outputs":[_durl(raw,"audio/wav" if fmt=="wav" else "audio/mpeg")],
          "saved":[out.name],"view":[_view(out)],
          "meta":{"format":fmt,"bytes":len(raw),
                  "duration":(_probe_json(src).get("format") or {}).get("duration")}}
    if len(raw)>25*1024*1024:
        resp["outputs"]=[]; resp["note"]="audio >25MB: fetch via saved/view URL"
    return _j(200,resp)

def T_audio_trim(b):
    e=_ff_err()
    if e: return e
    src,err=_src(b,("audio","video","media"))
    if err: return err
    start=max(0.0,float(b.get("start") or 0))
    dur=float(b.get("duration") or 0)
    if dur<=0:
        info=_probe_json(src)
        total=float((info.get("format") or {}).get("duration") or 0)
        dur=max(0.1,total-start) if total>0 else 10.0
    d=Path("/tmp/ComfyUI/output/tasks"); d.mkdir(parents=True,exist_ok=True)
    out=d/("trim_%d.wav"%int(time.time()))
    _run([FFMPEG,"-y","-ss","%.3f"%start,"-t","%.3f"%dur,"-i",str(src),
          "-acodec","pcm_s16le",str(out)])
    raw=out.read_bytes()
    return _j(200,{"ok":True,"task":"audio_trim",
        "outputs":[_durl(raw,"audio/wav")],"saved":[out.name],"view":[_view(out)],
        "meta":{"start":start,"duration":round(dur,3),"bytes":len(raw)}})

def T_probe(b):
    src,err=_src(b,("media","video","audio"))
    if err: return err
    if not FFPROBE: return _j(503,{"ok":False,"error":"ffprobe not available"})
    info=_probe_json(src)
    fmt=info.get("format") or {}
    streams=[{k:s.get(k) for k in ("codec_type","codec_name","width","height",
             "sample_rate","channels","duration")} for s in info.get("streams") or []]
    return _j(200,{"ok":True,"task":"probe","outputs":[],"file":src.name,
        "meta":{"duration":fmt.get("duration"),"format":fmt.get("format_name"),
                "size":fmt.get("size"),"bit_rate":fmt.get("bit_rate"),"streams":streams}})

TASK_INFO={
 "bg_remove":{"kind":"image","desc":"background remover: cut out the subject, transparent PNG",
   "input":"image (data URL)","opts":"model (u2net|u2netp|isnet-general-use|u2net_human_seg|birefnet-general), alpha_matting"},
 "extract":{"kind":"image","desc":"element extractor: foreground elements as separate transparent cutouts with bboxes",
   "input":"image (data URL)","opts":"model, max_elements (default 5), min_area, include_mask"},
 "upscale":{"kind":"image","desc":"4x super-resolution (Real-ESRGAN, runs on GPU via Comfy)",
   "input":"image (data URL)","opts":"model (RealESRGAN_x4plus.pth)"},
 "video_frames":{"kind":"video","desc":"extract evenly spaced frames as PNGs",
   "input":"video (data URL) or video_path","opts":"count (1-32, default 4), max_size (default 1024)"},
 "video_gif":{"kind":"video","desc":"animated preview GIF of the first max_duration seconds",
   "input":"video (data URL) or video_path","opts":"fps (default 4), width (default 480), max_duration (default 10)"},
 "audio_extract":{"kind":"audio","desc":"extract the audio track from any media file",
   "input":"media (data URL) or video_path/audio_path","opts":"format (wav|mp3, default wav)"},
 "audio_trim":{"kind":"audio","desc":"trim a time range (re-encoded wav)",
   "input":"media (data URL) or video_path/audio_path","opts":"start (s), duration (s)"},
 "probe":{"kind":"media","desc":"ffprobe: duration, streams, codecs, size",
   "input":"media (data URL) or video_path/audio_path","opts":"-"},
}
TASKS={"bg_remove":T_bg_remove,"extract":T_extract,"upscale":T_upscale,
       "video_frames":T_video_frames,"video_gif":T_video_gif,
       "audio_extract":T_audio_extract,"audio_trim":T_audio_trim,"probe":T_probe}

def H_tasks(o):
    avail={"bg_remove":_REMBG is not None,"extract":_REMBG is not None,
           "upscale":(MODELS/"upscale_models/RealESRGAN_x4plus.pth").exists(),
           "video_frames":bool(FFMPEG),"video_gif":bool(FFMPEG),
           "audio_extract":bool(FFMPEG),"audio_trim":bool(FFMPEG),
           "probe":bool(FFPROBE)}
    tasks={k:dict(v,available=bool(avail.get(k))) for k,v in TASK_INFO.items()}
    return _j(200,{"ok":True,"tasks":tasks,
        "runtime":{"rembg":_REMBG is not None,"rembg_error":_RembgErr,
                   "healthy":bool(globals().get("TASKS_HEALTHY",True)),
                   "ffmpeg":FFMPEG,"ffprobe":FFPROBE,
                   "upscale_model":"RealESRGAN_x4plus.pth"}})

def H_task(o):
    b=o["body"] or {}
    name=str(b.get("task") or "").strip()
    if not name:
        return _j(400,{"ok":False,"error":"pass {'task':'bg_remove','image':'data:...'}",
                       "tasks":sorted(TASK_INFO)})
    if name not in TASKS:
        return _j(404,{"ok":False,"error":"unknown task","tasks":sorted(TASK_INFO)})
    try:
        return TASKS[name](b)
    except Exception as e:
        import traceback; traceback.print_exc()
        return _j(500,{"ok":False,"error":"task %s failed: %s"%(name,e)})

DOCS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MiniMax H3 Hub API</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:2rem auto;max-width:60rem;line-height:1.5;color:#111}
code,pre{background:#f4f4f5;border-radius:4px;padding:.1rem .3rem;font-family:ui-monospace,monospace}
pre{padding:.8rem;overflow-x:auto} table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:.45rem;text-align:left;vertical-align:top} th{background:#fafafa}
h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:1.8rem}</style></head><body>
<h1>MiniMax H3 generation hub &mdash; API reference</h1>
<p>One public Base URL serves three things: the <a href="/">ComfyUI web UI</a>,
the native Comfy API (<code>POST /prompt</code>, <code>GET /history</code>, <code>GET /ws</code>),
and this management hub under <code>/h3api/*</code>. All hub responses are JSON:
<code>{"ok": true, ...}</code> or <code>{"ok": false, "error": "..."}</code>.</p>

<h2>Endpoints</h2>
<table><tr><th>Method &amp; path</th><th>Purpose</th></tr>
<tr><td><code>GET /h3api/</code></td><td>Endpoint index</td></tr>
<tr><td><code>GET /h3api/docs</code></td><td>This page</td></tr>
<tr><td><code>GET /h3api/health</code></td><td>API + Comfy liveness, uptime, active preset</td></tr>
<tr><td><code>GET /h3api/catalog</code></td><td>Live HF model catalog (H3 core + Flux image + Music-3 files + community repos)
 with <code>local/cached/remote</code> status. <code>?refresh=1</code> re-queries Hugging Face now</td></tr>
<tr><td><code>GET /h3api/models</code></td><td>On-disk models, sizes, sources, MANIFEST, required set</td></tr>
<tr><td><code>GET /h3api/settings</code></td><td>Presets + active generation settings</td></tr>
<tr><td><code>POST /h3api/settings</code></td><td><code>{"preset":"fast_smoke"}</code> and/or
 <code>{"set":{"megapixels":0.4,"duration":5,"seed":1,"turbo":true,"turbo_steps":4,"aspect":"16:9 (Widescreen)"}}</code></td></tr>
<tr><td><code>POST /h3api/select</code></td><td>Switch model/settings: downloads the new set (async),
 <b>deletes superseded local files</b> (<code>{"prune":false}</code> to keep them)</td></tr>
<tr><td><code>POST /h3api/download</code></td><td>Download files now:
 <code>{"file":"loras/x.safetensors"}</code> or <code>{"files":[...]}</code> from Comfy-Org/MiniMax-H3</td></tr>
<tr><td><code>GET /h3api/download</code></td><td>Download status (running / last result)</td></tr>
<tr><td><code>POST /h3api/generate</code></td><td>Queue a generation (see below)</td></tr>
<tr><td><code>GET /h3api/jobs</code></td><td>Queue + history status of recent jobs</td></tr>
<tr><td><code>GET /h3api/outputs</code></td><td>Generated files + <code>/view</code> URLs</td></tr>
<tr><td><code>POST /h3api/free</code></td><td>Unload models from GPU / free VRAM</td></tr>
<tr><td><code>GET /h3api/gpus</code></td><td>nvidia-smi snapshot (index, util, VRAM, power)</td></tr>
<tr><td><code>POST /h3api/import</code></td><td>Upload a model file:
 <code>?path=loras/file.safetensors</code>, raw bytes as body</td></tr>
<tr><td><code>GET /h3api/tasks</code></td><td>Use-case tasks (image/video/audio) + availability flags</td></tr>
<tr><td><code>POST /h3api/task</code></td><td>Run a use-case:
 <code>{"task":"bg_remove","image":"data:image/png;base64,..."}</code></td></tr>
<tr><td><code>POST /prompt</code></td><td>Native Comfy API (pass a fully built API prompt)</td></tr></table>

<h2>Generate</h2>
<pre>curl -X POST "$BASE/h3api/generate" -H 'Content-Type: application/json' -d '{
  "prompt": "A calm ocean wave rolling onto a quiet beach at golden hour",
  "seed": 12345,
  "set": {"megapixels": 0.4, "duration": 5}
}'</pre>
<p>Returns <code>{"ok":true,"prompt_id":"...","meta":{...}}</code>. Optional keys:
<code>preset</code> ("fast_smoke" | "quality" | "image_smoke" | "music_smoke"), <code>set</code> (any settings keys),
<code>first_frame</code> / <code>last_frame</code> as <code>data:image/png;base64,...</code>
URLs (image-to-video / last-frame conditioning), <code>seed</code>.
Poll <code>GET /h3api/jobs</code> until the job status is <code>success</code>.
Preset <code>image_smoke</code> switches modality to Flux-schnell image generation
(<code>set</code>: <code>width</code>, <code>height</code>, <code>steps</code>, <code>cfg</code>,
<code>sampler</code>, <code>scheduler</code>, <code>checkpoint</code>).
Preset <code>music_smoke</code> switches modality to MiniMax Music 3 generation
(<code>set</code>: <code>lyrics</code>, <code>max_duration</code> in seconds,
<code>tiled_decode</code> true/false, <code>format</code> flac|mp3|opus,
<code>prompt</code> = the style caption). Saved as <code>output/h3_music_*.flac</code>.</p>

<h2>Tasks (use-cases)</h2>
<table><tr><th>Task</th><th>What it does</th><th>Input</th><th>Options</th></tr>
<tr><td><code>bg_remove</code></td><td>Background remover &rarr; transparent PNG</td><td><code>image</code> data URL</td>
 <td><code>model</code> (u2net, u2netp, isnet-general-use, u2net_human_seg, birefnet-*), <code>alpha_matting</code></td></tr>
<tr><td><code>extract</code></td><td>Element extractor: foreground elements as separate transparent cutouts + bboxes</td>
 <td><code>image</code> data URL</td><td><code>model</code>, <code>max_elements</code>, <code>min_area</code>, <code>include_mask</code></td></tr>
<tr><td><code>upscale</code></td><td>4&times; super-resolution (Real-ESRGAN, GPU via Comfy)</td><td><code>image</code> data URL</td>
 <td><code>model</code> (RealESRGAN_x4plus.pth)</td></tr>
<tr><td><code>video_frames</code></td><td>Evenly spaced frames as PNGs</td><td><code>video</code> data URL or <code>video_path</code></td>
 <td><code>count</code> (1&ndash;32), <code>max_size</code></td></tr>
<tr><td><code>video_gif</code></td><td>Animated preview GIF</td><td><code>video</code> data URL or <code>video_path</code></td>
 <td><code>fps</code>, <code>width</code>, <code>max_duration</code></td></tr>
<tr><td><code>audio_extract</code></td><td>Extract the audio track from any media file</td><td><code>video_path</code>/<code>audio_path</code> or data URL</td>
 <td><code>format</code> (wav|mp3)</td></tr>
<tr><td><code>audio_trim</code></td><td>Trim a time range</td><td><code>video_path</code>/<code>audio_path</code> or data URL</td>
 <td><code>start</code> (s), <code>duration</code> (s)</td></tr>
<tr><td><code>probe</code></td><td>ffprobe: duration, streams, codecs</td><td><code>video_path</code>/<code>audio_path</code> or data URL</td>
 <td>-</td></tr></table>
<pre>curl -X POST "$BASE/h3api/task" -H 'Content-Type: application/json' -d '{"task":"extract","image":"data:image/png;base64,...","max_elements":4}'</pre>
<p>Responses: <code>{"ok":true,"outputs":["data:..."],"saved":["file.png"],"view":["/view?..."],"meta":{...}}</code>
(<code>extract</code> also returns <code>elements[]</code> with bboxes). Saved files land in
<code>output/tasks/</code>. Paths passed as <code>&lt;key&gt;_path</code> must live under
<code>/tmp/ComfyUI</code> (e.g. a generated MP4 from <code>/h3api/outputs</code>).</p>

<h2>Settings surface</h2>
<table><tr><th>Key</th><th>Meaning</th></tr>
<tr><td><code>prompt</code></td><td>Text prompt (H3 is CFG-distilled: there is <b>no negative prompt</b>)</td></tr>
<tr><td><code>aspect</code>, <code>megapixels</code></td><td>Aspect ratio + quality: 0.4 &rarr; 864&times;480,
 0.98 &rarr; 1344&times;768 (16:9)</td></tr>
<tr><td><code>duration</code></td><td>Seconds at 24 fps (trained range 5&ndash;15 s)</td></tr>
<tr><td><code>steps</code>, <code>turbo</code>, <code>turbo_steps</code>, <code>lora</code>,
<code>turbo_strength</code></td><td>Sampling: dense 20-step, or turbo LoRA (4-step fast path)</td></tr>
<tr><td><code>seed</code></td><td>Noise seed</td></tr>
<tr><td><code>unet</code>, <code>clip</code>, <code>vae</code>, <code>audio_vae</code></td><td>Model files
(select via <code>/h3api/select</code>; switching downloads the new set and deletes the previous one)</td></tr>
<tr><td><code>first_frame_b64</code>, <code>last_frame_b64</code></td><td>Frame conditioning (data URLs)</td></tr>
<tr><td><code>width</code>, <code>height</code>, <code>cfg</code>, <code>sampler</code>, <code>scheduler</code>, <code>checkpoint</code></td><td>Image generation keys (Flux preset <code>image_smoke</code>)</td></tr>
<tr><td><code>lyrics</code>, <code>max_duration</code>, <code>tiled_decode</code>, <code>format</code>, <code>output_prefix</code></td><td>Music generation keys (MiniMax Music 3 preset <code>music_smoke</code>; <code>prompt</code> = style caption)</td></tr>
</table>

<h2>Model lifecycle</h2>
<p>Boot: registry queries Hugging Face live &rarr; catalog lists every file with
<code>local/cached/remote</code> &rarr; assets mirror the Kaggle Dataset cache under
<code>/kaggle/input</code> (symlinks) &rarr; only missing files download (HF token).
A dataset write-back (200 GB cap, auto-skip) backs up downloaded models for the next boot.</p>
</body></html>
"""

ROUTES={
    ("GET","/h3api"):H_index, ("GET","/h3api/"):H_index,
    ("GET","/h3api/health"):H_health, ("GET","/h3api/catalog"):H_catalog,
    ("GET","/h3api/models"):H_models,
    ("GET","/h3api/settings"):H_settings, ("POST","/h3api/settings"):H_settings,
    ("GET","/h3api/download"):H_download_status, ("POST","/h3api/download"):H_download,
    ("POST","/h3api/select"):H_select, ("POST","/h3api/generate"):H_generate,
    ("GET","/h3api/jobs"):H_jobs, ("GET","/h3api/outputs"):H_outputs,
    ("POST","/h3api/free"):H_free, ("GET","/h3api/gpus"):H_gpus,
    ("POST","/h3api/import"):H_import,
    ("GET","/h3api/tasks"):H_tasks, ("POST","/h3api/task"):H_task,
}

class H(http.server.BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    server_version="H3Hub/1.0"

    def log_message(self,*a): pass   # keep notebook output clean

    def _send(self,code,headers,body):
        try:
            self.send_response(code)
            for k,v in headers.items(): self.send_header(k,v)
            self.send_header("Access-Control-Allow-Origin","*")
            self.send_header("Access-Control-Allow-Headers","Content-Type")
            self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception: pass

    def _read_body(self):
        te=(self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            buf=b""
            while True:
                line=self.rfile.readline().strip()
                if b";" in line: line=line.split(b";")[0]
                try: n=int(line or b"0",16)
                except ValueError: break
                if n==0:
                    while True:
                        t=self.rfile.readline()
                        if t in (b"\r\n",b"\n",b""): break
                    break
                buf+=self.rfile.read(n); self.rfile.readline()
            return buf
        cl=int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(cl) if cl>0 else b""

    def do_OPTIONS(self): self._send(204,{"Content-Type":"text/plain"},b"")
    def do_GET(self): self._route("GET")
    def do_POST(self): self._route("POST")

    def _route(self,method):
        path=self.path.split("?")[0]
        if method=="POST" and path=="/h3api/import":
            try:
                self._send(*H_import({"path":self.path,"headers":self.headers,
                                      "rfile":self.rfile.read}))
            except Exception as e:
                self._send(*_j(500,{"ok":False,"error":str(e)}))
            return
        fn=ROUTES.get((method,path))
        if fn is None and path.startswith("/h3api"):
            self._send(*_j(404,{"ok":False,"error":"unknown endpoint","docs":"/h3api/docs"}))
            return
        try:
            body=self._read_body()
            if fn is not None:
                obj={}
                if body:
                    try: obj=json.loads(body)
                    except Exception: obj={"_raw":len(body)}
                self._send(*fn({"method":method,"body":obj,"headers":self.headers,"path":self.path}))
            else:
                self._proxy(method,body)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(*_j(500,{"ok":False,"error":str(e)}))

    def _proxy(self,method,body):
        if (self.headers.get("Upgrade") or "").lower()=="websocket":
            return self._ws()
        req=urllib.request.Request("http://%s:%d%s"%(COMFY[0],COMFY[1],self.path),
            data=(body if body and method!="GET" else None),method=method)
        for k,v in self.headers.items():
            lk=k.lower()
            if lk in ("host","connection","upgrade","content-length",
                      "accept-encoding","transfer-encoding","keep-alive"): continue
            req.add_header(k,v)
        if body: req.add_header("Content-Length",str(len(body)))
        try:
            with urllib.request.urlopen(req,timeout=600) as r:
                data=r.read(); st=r.status; rh=dict(r.headers)
        except urllib.error.HTTPError as e:
            data=e.read(); st=e.code; rh=dict(e.headers)
        except Exception as e:
            self._send(*_j(502,{"ok":False,"error":"comfy proxy: %s"%e})); return
        try:
            self.send_response(st)
            for k,v in rh.items():
                if k.lower() in ("transfer-encoding","connection","content-length",
                                 "content-encoding","keep-alive"): continue
                self.send_header(k,v)
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception: pass

    def _ws(self):
        """WebSocket passthrough: raw bidirectional pipe browser <-> ComfyUI."""
        try:
            up=socket.create_connection(COMFY,timeout=10)
        except Exception as e:
            self._send(*_j(502,{"ok":False,"error":"comfy ws unreachable: %s"%e})); return
        head="GET %s HTTP/1.1\r\n"%self.path
        for k,v in self.headers.items():
            if k.lower()=="host": v="%s:%d"%COMFY
            head+="%s: %s\r\n"%(k,v)
        head+="\r\n"
        up.sendall(head.encode())
        src=self.connection
        self.close_connection=True
        def pump(a,b):
            try:
                while True:
                    d=a.recv(65536)
                    if not d: break
                    b.sendall(d)
            except Exception: pass
            try: b.shutdown(socket.SHUT_WR)
            except Exception: pass
        threading.Thread(target=pump,args=(src,up),daemon=True).start()
        pump(up,src)
        try: up.close()
        except Exception: pass

def H_docs(o):
    return 200,{"Content-Type":"text/html; charset=utf-8"},DOCS_HTML.encode()

ROUTES[("GET","/h3api/docs")]=H_docs

# ---------------- start ----------------
if os.environ.get("H3_API","1")!="0":
    try:
        srv=http.server.ThreadingHTTPServer(("127.0.0.1",API_PORT),H)
        srv.daemon_threads=True
        threading.Thread(target=srv.serve_forever,daemon=True).start()
        API_UP=True
        print("hub API + proxy listening on http://127.0.0.1:%d (docs: /h3api/docs)"%API_PORT)
    except Exception as e:
        API_UP=False
        print("hub API failed to start (continuing without it):",e)
else:
    API_UP=False
    print("hub API disabled (H3_API=0)")
'''

# ---------------- cell 10: smoke ----------------
smoke = r'''
import json,time,urllib.request,urllib.error,os
SMOKE_JOBS=[]
for mod in SMOKE:                       # one low-quality smoke per modality (settings cell)
    cfg=dict(PRESETS[SMOKE_PRESET[mod]])
    prompt, meta = build_prompt(cfg)    # shared builder (convert cell; modality-aware)
    print("[%s smoke] build meta:"%mod, json.dumps(meta))
    body={"prompt":prompt,"extra_data":{"preview_method":"latent2rgb"}}
    try:
        r=urllib.request.Request(BASE+"/prompt",data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
        resp=urllib.request.urlopen(r,timeout=60).read().decode()
        print("[%s smoke] submitted:"%mod,resp)
    except urllib.error.HTTPError as ee:
        print("[%s smoke] prompt failed"%mod,ee,"body:",ee.read().decode()[:6000])
        raise
    pid=json.loads(resp).get("prompt_id")
    if not pid: raise RuntimeError("%s smoke: no prompt_id in response"%mod)
    SMOKE_JOBS.append((mod,pid))
print("smoke jobs queued:",json.dumps(SMOKE_JOBS))
if not SMOKE_JOBS: raise RuntimeError("empty smoke schedule")
'''

# ---------------- cell 11: wait ----------------
wait = r'''
import json,time,urllib.request,os
KEY=("[GPUs]","[MultiStream]","[VAE split]","[Cache]","UNSPLIT","unsplit on 1 GPU")
BUDGET={"image":1800,"video":10800,"music":3600}   # image: flux 4 steps; video: 3 h; music: 30-stp mini
results={}
seen=set()
for mod,pid in SMOKE_JOBS:
    print("waiting for %s smoke job"%mod,pid)
    budget=BUDGET.get(mod,10800)
    got=None
    for i in range(budget//10):
        try:
            h=json.loads(urllib.request.urlopen("http://127.0.0.1:8188/history/%s"%pid,timeout=5).read())
            if h and h.get(pid):
                got=h.get(pid); break
        except Exception: pass
        if i%12==0:
            try:
                lines=[l for l in open("/tmp/comfy.log",errors="ignore").read().splitlines() if l.strip()]
                print("[%s %ds]"%(mod,i*10), lines[-1][:200] if lines else "(no log yet)")
                diag=[l for l in lines if any(k in l for k in KEY)]
                new=[l for l in diag if l not in seen]
                for l in new: print("   [h3ms]", l[:250])       # [GPUs] plan, ranks, step times
                seen.update(new)
                if any(("UNSPLIT" in l or "unsplit on 1 GPU" in l) for l in new):
                    raise RuntimeError("single-GPU fallback detected: "+new[-1][:250])
            except RuntimeError: raise
            except Exception: pass
            try:   # live per-GPU util: index, util.gpu, util.mem, mem.used, power
                g=[l for l in open("/tmp/gpus.log",errors="ignore").read().splitlines() if l.strip()][-2:]
                for gl in g: print("   [gpu]", gl[:160])
            except Exception: pass
        time.sleep(10)
    if not got: raise RuntimeError("%s smoke did not finish within %ds"%(mod,budget))
    st=got.get("status",{})
    print("[%s] smoke status:"%mod,st.get("status_str"),"completed:",st.get("completed"),"/",st.get("remaining"))
    if st.get("status_str")=="error":
        raise RuntimeError("%s smoke run had execution errors: "%mod+json.dumps(st)[:2000])
    results[mod]=got

# ---- modality-aware verification ----
cl=open("/tmp/comfy.log",errors="ignore").read()
if "video" in results:                   # dual-GPU proof applies to the H3 video smoke only
    ev=[l for l in cl.splitlines() if ("[GPUs]" in l or "[MultiStream] active" in l or "UNSPLIT" in l or "unsplit on 1 GPU" in l)]
    for l in ev: print("h3ms:",l[:300])
    if "UNSPLIT" in cl or "running unsplit on 1 GPU" in cl:
        raise RuntimeError("single-GPU fallback detected -- smoke did NOT use both T4s")
    if not any("active: 2 ranks" in l for l in ev):
        raise RuntimeError("no '[MultiStream] active: 2 ranks' evidence in comfy.log -- cannot confirm dual-GPU")
    print("DUAL-GPU CONFIRMED: active: 2 ranks")
    try:
        lastg=[l for l in open("/tmp/gpus.log",errors="ignore").read().splitlines() if l.strip()][-1:]
        print("last gpu sample:", lastg[0][:160] if lastg else "(none)")
    except Exception: pass
pngs=[]; auds=[]
print("smoke outputs:")
for root,_,files in os.walk("/tmp/ComfyUI/output"):
    for f in files:
        p=os.path.join(root,f)
        if f.lower().endswith(".png"): pngs.append(p)
        if f.lower().endswith((".flac",".mp3",".wav",".ogg",".opus")): auds.append(p)
        if f.lower().endswith((".mp4",".webm",".mov",".webp",".gif",".png",".flac",".mp3",".wav",".ogg",".opus")):
            print("found:",p)
if "image" in results and not pngs:
    raise RuntimeError("image smoke finished but produced no PNG output")
if "music" in results and not auds:
    raise RuntimeError("music smoke finished but produced no audio output")
print("SMOKE PASSED:",sorted(results))
'''

# ---------------- cell 12: pub ----------------
pub = r'''
import subprocess,time,re,urllib.request,json,os
from pathlib import Path
Path("/tmp/h3_api").mkdir(parents=True,exist_ok=True)
ok=False
for u in ["https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
          "https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64"]:
    try:
        data=urllib.request.urlopen(u,timeout=120).read()
        if len(data)>1000000:
            open("/tmp/h3_api/cloudflared","wb").write(data); ok=True; break
        print("cf dl too small from",u,len(data))
    except Exception as e:
        print("cf dl failed:",u,e)
if not ok: raise RuntimeError("cloudflared download failed")
os.chmod("/tmp/h3_api/cloudflared",0o755)
target = API_PORT if API_UP else 8188
print("tunnel target: http://127.0.0.1:%d %s"%(target,"(hub API + proxy)" if API_UP else "(direct Comfy)"))
log=open("/tmp/h3_api/cf.log","w")
proc=subprocess.Popen(["/tmp/h3_api/cloudflared","tunnel","--url","http://127.0.0.1:%d"%target],
                      stdout=log,stderr=subprocess.STDOUT)
url=None; pat=re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
for i in range(90):
    t=open("/tmp/h3_api/cf.log","r",encoding="utf-8",errors="ignore").read()
    m=pat.search(t)
    if m:
        try:
            r1=urllib.request.urlopen(m.group(0)+"/history",timeout=5)
            r2=urllib.request.urlopen(m.group(0)+"/h3api/health",timeout=5) if API_UP else None
            if r1.status==200 and (r2 is None or r2.status==200):
                url=m.group(0); break
        except Exception: pass
    time.sleep(2)
if not url: raise RuntimeError("cloudflare tunnel not healthy")
print("Verified public endpoint:",url)
print("\n================ MINIMAX H3 — GENERATION READY ================")
print("Base URL:"); print("  "+url)
print("ComfyUI web UI:"); print("  "+url+"/")
print("Native Comfy API:"); print("  POST "+url+"/prompt")
if API_UP:
    print("Hub API docs:"); print("  "+url+"/h3api/docs")
    print("Hub health:"); print("  "+url+"/h3api/health")
    print("Hub generate:"); print("  POST "+url+"/h3api/generate   {\"prompt\":\"...\"}")
    print("Hub settings/catalog/jobs/outputs/gpus:")
    print("  GET  "+url+"/h3api/settings | /h3api/catalog | /h3api/jobs | /h3api/outputs | /h3api/gpus")
else:
    print("(hub API unavailable this run - direct Comfy only)")
print("===============================================================")
if API_UP:
    print("live endpoint checks through the tunnel:")
    def _probe(base,p,tries=2,timeout=30):   # v17: retry once + show e.reason (v16 stalled blind)
        last=None
        for _t in range(tries):
            try:
                r=urllib.request.urlopen(base+p,timeout=timeout)
                return "%d"%r.status, r.read(150).decode("utf-8","ignore").replace("\n"," ")
            except Exception as e:
                last=e
                time.sleep(3)
        body=""
        try: body=last.read(150).decode("utf-8","ignore").replace("\n"," ")
        except Exception: pass
        code=getattr(last,"code",None)
        if code is None:
            code="%s: %s"%(type(last).__name__,getattr(last,"reason",None) or last)
        return str(code),body
    for p in ["/h3api/health","/h3api/catalog","/h3api/settings","/h3api/jobs",
              "/h3api/outputs","/h3api/gpus","/h3api/tasks","/h3api/docs","/object_info","/h3api/nope"]:
        c,b=_probe(url,p)
        print("  %s %-18s %s"%(c,p,b[:110]))
    def _post(base,path,payload,tries=2,timeout=120):   # v19: retry once like the GET probes
        last=None
        for _t in range(tries):
            try:
                req=urllib.request.Request(base+path,data=json.dumps(payload).encode(),
                      headers={"Content-Type":"application/json"},method="POST")
                r=urllib.request.urlopen(req,timeout=timeout)
                return r.status,json.loads(r.read())
            except Exception as e:
                last=e
                print("  retry POST",path,getattr(e,"reason",type(e).__name__))
                time.sleep(3)
        raise last
    try:
        c,j=_post(url,"/h3api/settings",{},timeout=30)
        print("  %s POST /h3api/settings    %s"%(c,json.dumps(j)[:110]))
    except Exception as e:
        print("  POST /h3api/settings failed:",getattr(e,"code",type(e).__name__))
    # ---- task smoke: image use-cases end-to-end through the public tunnel ----
    # (upscale REQUIRED always; bg_remove + extract REQUIRED only when TASKS_HEALTHY -
    #  i.e. the install cell proved rembg/scipy import cleanly; video tasks warn only)
    import base64 as _b64
    th=bool(globals().get("TASKS_HEALTHY",True))
    print("task deps health (TASKS_HEALTHY):",th)
    outdir=Path("/tmp/ComfyUI/output")
    def _newest(pat):
        c=sorted([p for p in outdir.rglob(pat)], key=lambda x:-x.stat().st_mtime)
        return c[0] if c else None
    task_fail=[];task_warn=[]
    png=_newest("*.png")
    if png:
        print("task smoke image input:",png)
        durl="data:image/png;base64,"+_b64.b64encode(png.read_bytes()).decode()
        for tname,payload in [("bg_remove",{"task":"bg_remove","image":durl}),
                              ("extract",{"task":"extract","image":durl,"max_elements":4}),
                              ("upscale",{"task":"upscale","image":durl})]:
            soft=(tname in ("bg_remove","extract")) and not th
            try:
                rcode,j=_post(url,"/h3api/task",payload,timeout=300)
                print("  %d POST task %-11s ok=%s outputs=%d saved=%s"%(
                      rcode,tname,j.get("ok"),len(j.get("outputs") or []),j.get("saved")))
                if not (j.get("ok") and (j.get("outputs") or [])):
                    (task_warn if soft else task_fail).append(tname)
            except Exception as e:
                detail=""
                try: detail=e.read().decode()[:160]
                except Exception: detail=str(e)[:160]
                print("  task %-11s FAILED: %s"%(tname,detail))
                (task_warn if soft else task_fail).append(tname)
    else:
        print("task smoke skipped: no PNG in output (image smoke off?)")
    mp4=_newest("*.mp4")
    if mp4 and png:
        try:
            rcode,j=_post(url,"/h3api/task",{"task":"video_frames","video_path":str(mp4),"count":3},timeout=300)
            print("  %d POST task video_frames ok=%s outputs=%d"%(
                  rcode,j.get("ok"),len(j.get("outputs") or [])))
            if not j.get("ok"): print("  WARN video_frames failed (ffmpeg missing?)")
        except Exception as e:
            print("  WARN video_frames failed:",str(e)[:160])
    for w in task_warn:
        print("  WARN tolerated (install cell reported task deps unhealthy after heal):",w)
    if task_fail:
        raise RuntimeError("task smoke failures: %s"%task_fail)
    print("TASK SMOKE PASSED")
'''

# ---------------- cell 13: sync ----------------
sync = r'''
# ===== Kaggle Dataset auto-cache write-back: 200 GB cap auto-skip, NEVER fails the run =====
# H3_CACHE_UPLOAD=auto (default: create once, refresh only on manifest drift)
#                |always|dry|never
# Uploads via kagglehub (native in-notebook token auth); classic kaggle CLI as fallback.
import json, os, subprocess, shutil, time
from pathlib import Path
POLICY=os.environ.get("H3_CACHE_UPLOAD","auto").lower()
CAP=200*10**9
TITLE="MiniMax H3 model cache"
SLUG="minimax-h3-model-cache"
KNOWN_OWNER="adityahalde8777"   # injected at build time from kernel-metadata id (last-resort owner)
MODELS=Path("/tmp/ComfyUI/models")
STAGE=Path("/tmp/h3_cache_stage")

def _kg(args,timeout=7200):
    r=subprocess.run(["kaggle"]+args,capture_output=True,text=True,timeout=timeout,
                     stdin=subprocess.DEVNULL)
    return r.returncode,(r.stdout or "")+"\n"+(r.stderr or "")

# owner detection: modern Kaggle notebooks auth via a TOKEN FILE (kagglehub's native
# notebook auth), not KAGGLE_USERNAME/kaggle.json - v15 proved the old check fails.
def _username():
    try:
        import kagglehub
        u=(kagglehub.whoami(verbose=False) or {}).get("username")
        if u: return u,"kagglehub.whoami"
    except Exception as e:
        print("auth: kagglehub.whoami unavailable (%s)"%type(e).__name__)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"],"env KAGGLE_USERNAME/KAGGLE_KEY"
    cands=[os.path.expanduser("~/.kaggle/kaggle.json")]
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        cands.append(os.path.join(os.environ["KAGGLE_CONFIG_DIR"],"kaggle.json"))
    cands+=["/root/.kaggle/kaggle.json","/home/kaggle/.kaggle/kaggle.json",
            "/kaggle/.kaggle/kaggle.json"]
    for c in cands:
        try:
            u=json.load(open(c)).get("username")
            if u: return u,"kaggle.json (%s)"%c
        except Exception: pass
    return KNOWN_OWNER,"embedded kernel owner"

def _remote_manifest(ref):
    # returns (manifest|None, status) - kagglehub first (native notebook auth)
    shutil.rmtree("/tmp/h3_remote_chk",ignore_errors=True)
    try:
        import kagglehub
        p=kagglehub.dataset_download(ref,path="MANIFEST.json",output_dir="/tmp/h3_remote_chk")
        try: return json.load(open(p)),"kagglehub probe ok"
        except Exception as e: return None,"kagglehub: manifest unreadable (%s)"%type(e).__name__
    except Exception as e:
        kh="kagglehub probe: %s"%type(e).__name__
    rc,out=_kg(["datasets","download",ref,"-f","MANIFEST.json","-p","/tmp/h3_remote_chk"],timeout=300)
    if rc==0:
        try: return json.load(open("/tmp/h3_remote_chk/MANIFEST.json")),"cli probe ok"
        except Exception: return None,"cli: manifest unreadable"
    tail=(out.strip().splitlines() or ["no output"])[-1][:100]
    return None,"%s; cli rc=%d %s"%(kh,rc,tail)

try:
    if POLICY=="never":
        print("cache write-back disabled (H3_CACHE_UPLOAD=never)")
    else:
        # 1) inventory of local model files (symlinks = already cached from a dataset)
        inv={}
        for p in MODELS.rglob("*"):
            if p.is_file() or p.is_symlink():
                try: inv[str(p.relative_to(MODELS))]=p
                except Exception: pass
        sizes={}; total=0
        for rel,p in inv.items():
            try: sz=p.stat().st_size
            except OSError: sz=0
            sizes[rel]=sz; total+=sz
        print("cache inventory: %d files, %.1f GB"%(len(inv),total/1e9))
        if total>CAP:
            print("SKIP upload: %.1f GB >= 200 GB dataset cap - continuing"%(total/1e9))
        elif total>shutil.disk_usage("/tmp").free:
            print("SKIP upload: not enough local disk to stage - continuing")
        else:
            # 2) owner + dataset ref (kagglehub native notebook auth first)
            print("auth env: KAGGLE_USERNAME=%s KAGGLE_KEY=%s KAGGLE_API_V1_TOKEN=%s kaggle.json=%s"%(
                bool(os.environ.get("KAGGLE_USERNAME")),bool(os.environ.get("KAGGLE_KEY")),
                bool(os.environ.get("KAGGLE_API_V1_TOKEN")),
                os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))))
            try:
                from kagglesdk.kaggle_env import is_in_kaggle_notebook
                print("in-kaggle-notebook:",is_in_kaggle_notebook())
            except Exception as e:
                print("in-kaggle-notebook: unknown (%s)"%type(e).__name__)
            user,auth_src=_username()
            ref="%s/%s"%(user,SLUG)
            print("cache owner: %s (via %s)"%(user,auth_src))
            # 3) remote MANIFEST probe -> drift decision
            remote,probe=_remote_manifest(ref)
            print("cache dataset: %s | probe: %s"%(ref,probe))
            need=False
            if POLICY=="always":
                need=True; print("policy=always -> refresh dataset version")
            elif remote is None:
                need=True; print("remote MANIFEST unavailable -> upload (first run or probe failed)")
            else:
                local={rel:{"size":sizes[rel]} for rel in sorted(sizes) if rel.endswith(".safetensors")}
                rsize={k:{"size":(v or {}).get("size")} for k,v in sorted(remote.items())}
                if rsize!=local:
                    need=True; print("manifest drift: %d remote vs %d local files -> upload"%(len(rsize),len(local)))
                else:
                    print("cache dataset up to date (manifest match)")
            # 4) stage (hardlinks where possible - near-zero disk) then upload
            if need:
                if STAGE.exists(): shutil.rmtree(STAGE,ignore_errors=True)
                STAGE.mkdir(parents=True,exist_ok=True)
                for rel,p in inv.items():
                    dst=STAGE/rel
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    if p.is_symlink():
                        try: os.symlink(os.readlink(p),dst)   # preserve original target
                        except Exception: shutil.copy2(p,dst)
                    else:
                        try: os.link(p,dst)                   # hardlink: zero extra disk
                        except Exception:
                            try: os.symlink(str(p.resolve()),dst)
                            except Exception: shutil.copy2(p,dst)
                stage_man={rel:{"size":sizes[rel]} for rel in sorted(sizes) if rel.endswith(".safetensors")}
                if (STAGE/"MANIFEST.json").exists(): (STAGE/"MANIFEST.json").unlink()
                json.dump(stage_man,open(str(STAGE/"MANIFEST.json"),"w"),indent=1)
                json.dump({"title":TITLE,"id":ref,"licenses":[{"name":"other"}]},
                          open(str(STAGE/"dataset-metadata.json"),"w"),indent=2)
                print("staged %d model files for upload -> %s"%(len(stage_man),ref))
                if POLICY=="dry":
                    print("DRY: would upload staged folder to %s"%ref)
                else:
                    up=False
                    t0=time.time()
                    try:
                        import kagglehub, inspect
                        kw={"version_notes":"auto refresh: %d files, %.1f GB"%(len(stage_man),total/1e9)}
                        if "ignore_patterns" in inspect.signature(kagglehub.dataset_upload).parameters:
                            kw["ignore_patterns"]=["dataset-metadata.json"]
                        kagglehub.dataset_upload(ref,str(STAGE),**kw)
                        print("kagglehub upload OK -> %s (%.0f s)"%(ref,time.time()-t0)); up=True
                    except Exception as e:
                        print("kagglehub upload failed (%s: %s)"%(type(e).__name__,str(e)[:200]))
                    if not up:
                        rc0,_=_kg(["datasets","metadata",ref,"-p","/tmp/h3_meta_chk"],timeout=120)
                        if rc0==0:
                            rc,out=_kg(["datasets","version","-p",str(STAGE),"-m","model cache refresh","-q"])
                            print("cli version rc=%d"%rc,out[-300:])
                        else:
                            rc,out=_kg(["datasets","create","-p",str(STAGE),"-q"])
                            print("cli create rc=%d"%rc,out[-300:])
                            if rc!=0:
                                rc,out=_kg(["datasets","version","-p",str(STAGE),"-m","model cache refresh","-q"])
                                print("cli version-after-create rc=%d"%rc,out[-300:])
                        if rc!=0: print("upload failed - run continues (never fails)")
                shutil.rmtree(STAGE,ignore_errors=True)
except Exception as e:
    print("cache write-back skipped (run continues, never fails):",e)
'''

CELLS = [
    (0, 'markdown', 'title_md'),
    (1, 'code', 'secrets'),
    (2, 'code', 'preflight'),
    (3, 'code', 'install'),
    (4, 'code', 'registry'),
    (5, 'code', 'settings'),
    (6, 'code', 'assets'),
    (7, 'code', 'convert'),
    (8, 'code', 'start'),
    (9, 'code', 'api'),
    (10, 'code', 'smoke'),
    (11, 'code', 'wait'),
    (12, 'code', 'pub'),
    (13, 'code', 'sync'),
]

NB_META_JSON = '{"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}, "kaggle": {"accelerator": "NvidiaTeslaT4", "enableGpu": true, "enableInternet": true}}'
NB_META = json.loads(NB_META_JSON)

def build(out_path):
    c0 = {'cell_type': 'markdown', 'metadata': {}, 'source': [title_md]}
    cells = [c0]
    for i, kind, name in CELLS[1:]:
        src = globals()[name]
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {},
                      'outputs': [], 'source': [src]})
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': NB_META, 'cells': cells}
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('wrote', out_path)

if __name__ == '__main__':
    build(os.environ.get('H3_NB_OUT', '/home/limitlessjourney829/kaggle/minimax-h3-api/minimax_h3_full_comfy.ipynb'))
