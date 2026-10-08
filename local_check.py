#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# local_check.py — offline gate for minimax_h3_full_comfy.ipynb.
#
# Run BEFORE every `kaggle kernels push`:
#   python3 minimax-h3-api/local_check.py                 # structural (no deps)
#   ./ComfyUI/venv/bin/python minimax-h3-api/local_check.py --live http://127.0.0.1:8191
#
#   --live URL   fetch REAL object_info from a running local ComfyUI and POST each
#                flattened smoke prompt to /prompt; assert the only node_errors are
#                value_not_in_list on model-loader widgets (expected: no model files
#                exist locally). Needs an installed venv + running server.
#
# Checks (exit != 0 on any FAIL):
#   1. rebuild notebook from build_full.py + ast-parse every code cell
#   2. exec the `convert` cell and build the video / image / music smoke prompts
#      against `tests/objinfo_fixture.json` (or `--live` object_info)
#   3. structural asserts per lane (node counts, save-node widgets, H3MultiStream
#      splice, no dangling links, no MISSING sentinels)
#   4. (deep, when comfy_api/latest/_io is importable) DynamicCombo kwargs sim:
#      get_finalized_class_inputs + build_nested_inputs must deliver the EXACT dict
#      SaveAudioAdvanced.execute/SaveVideo.execute consume — the v21 bug class
#   5. (--live) real /prompt validation, filtering loader-name value_not_in_list
#
# Requires the workflow templates at /tmp/h3_t2v.json, /tmp/flux_schnell.json,
# /tmp/audio_minimax_music_3.json (fetch like the assets cell does).
# =============================================================================
import argparse, ast, json, os, re, sys, urllib.error, urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
FIXTURE = os.path.join(REPO, "tests", "objinfo_fixture.json")

ap = argparse.ArgumentParser()
ap.add_argument("--live", default=os.environ.get("H3_LIVE_COMFY", ""))
ap.add_argument("--comfy", default=os.environ.get("COMFYUI_DIR", ""),
                help="path to a ComfyUI clone (enables the DynamicCombo kwargs deep checks)")
ap.add_argument("--skip-deep", action="store_true")
args, _ = ap.parse_known_args()

if args.comfy and os.path.isdir(args.comfy):
    sys.path.insert(0, args.comfy)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except Exception:
            raise RuntimeError(f"HTTP {e.code} from {url}: {body[:300]}") from e


# ------------------------------------------------------------------ templates (auto-fetch like the assets cell)
TEMPLATES = {
    "/tmp/h3_t2v.json": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v.json",
    "/tmp/flux_schnell.json": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/flux_schnell.json",
    "/tmp/audio_minimax_music_3.json": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/audio_minimax_music_3.json",
}
import urllib.request as _ur
for _path, _url in TEMPLATES.items():
    if not os.path.exists(_path):
        print(f"  (info: fetching {os.path.basename(_path)} from workflow_templates)")
        _ur.urlretrieve(_url, _path)

# ------------------------------------------------------------------ 1. rebuild + ast
import build_full as bf  # noqa: E402

out = os.environ.get("H3_NB_OUT", os.path.join(REPO, "minimax_h3_full_comfy.ipynb"))
bf.build(out)
nb = json.load(open(out))
ast_ok = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
        ast_ok += 1
ncells = len([c for c in nb["cells"] if c["cell_type"] == "code"])
check("rebuild + ast all cells", ast_ok == ncells, f"{ast_ok}/{ncells} code cells")

# 1b. nbformat no-warnings baseline — v21/v21b died invisibly because cells had NO id
#     fields and the Kaggle runner had promoted nbformat's MissingIDFieldWarning to a
#     hard error (zero logs, zero artifacts). Every cell must carry a unique id and
#     nbformat.validate must pass with ZERO warnings before we even think about pushing.
import warnings
try:
    import nbformat
    _ids = [c.get("id") for c in nb["cells"]]
    _have = bool(_ids) and all(_ids) and len(set(_ids)) == len(_ids)
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        nbformat.validate(nb)
    _bad = [str(x.message)[:70] for x in _w]
    check("nbformat validate (no warnings)", _have and not _bad,
          f"{len(nb['cells'])} cells, ids unique, warnings={_bad or 'none'}")
except Exception as e:  # pragma: no cover
    check("nbformat validate (no warnings)", False, f"{type(e).__name__}: {e}")

# 1c. v22.1 guard: the faster-whisper install line must pin av in the 17..18 band, never
#     13/19. ComfyUI's requirements.txt declares av>=17.0.0 (it imports av in 6 comfy_extras
#     modules at boot) and fw 1.2.1 needs av<19 (metadata_errors kwarg) — the v22 av==13.1.0
#     pin violated ComfyUI's requirement in the server's own site-packages and the ComfyUI
#     process never listened. Match only quoted version literals, not prose comments.
def _cell_with(mark):
    for c in nb["cells"]:
        sc = "".join(c.get("source", []))
        if c["cell_type"] == "code" and mark in sc:
            return sc
    return ""
_ins_cell = _cell_with('mark("cell:install:start")')
_avs = re.findall(r'"av==([0-9]+(?:\.[0-9]+)*)"', _ins_cell)
_av_ok = bool(_avs) and all(int(v.split(".")[0]) in (17, 18) for v in _avs)
check("install: faster-whisper av pin in 17..18 (not 13/19)", _av_ok,
      f"av version literals: {_avs or 'none'}")

# 1d. v22.1 guard: the ComfyUI boot watchdog must be present so a boot failure prints the
#     /tmp/comfy.log tail + process exit code (the v23 run died silently for a full 363 s
#     window because the server's stderr went to a file). Heartbeat + tail + relaunch.
_st_cell = _cell_with('mark("cell:start:start")')
_wd_ok = all(k in _st_cell for k in ("comfy.log tail (final)",
                                     "boot t=%ds alive=yes",
                                     "relaunching Comfy once"))
check("start: Comfy boot watchdog present (heartbeat/tail/relaunch)", _wd_ok,
      f"start cell {len(_st_cell)} chars")

# ------------------------------------------------------------------ 2. exec convert cell
src = open(os.path.join(REPO, "build_full.py")).read()
m = re.search(r"^convert = r''' (.*?) '''$", src, re.S | re.M)
if not m:
    m = re.search(r"^convert = r'''(.*?)'''$", src, re.S | re.M)
assert m, "convert cell var not found"
ns = {"__name__": "convert_cell"}
exec(compile(m.group(1), "convert_cell", "exec"), ns)

if args.live:
    live_info = fetch(args.live.rstrip("/") + "/object_info")
    # core-only local install lacks the Comfy-H3-MultiStream pack; merge its schema
    # from the fixture so the video splice can still be BUILT locally.
    fix = json.load(open(FIXTURE))
    h3_pack_present = "H3MultiStream" in live_info
    if not h3_pack_present:
        live_info["H3MultiStream"] = fix["H3MultiStream"]
        print("  (info: H3MultiStream from fixture — pack not installed locally)")
    ns["load_info"] = lambda: live_info
    print(f"  (info: live object_info {len(live_info)} classes)")
else:
    ns["load_info"] = lambda: json.load(open(FIXTURE))

VIDEO_CFG = dict(modality="video", unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                 clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                 vae="minimax_h3_video_vae_int8_convrot.safetensors",
                 audio_vae="minimax_h3_audio_vae_fp32.safetensors",
                 lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                 prompt="test", aspect="16:9 (Widescreen)", megapixels=0.4, duration=5,
                 seed=12345, turbo=True, turbo_steps=4, turbo_strength=1.0, output_prefix="h3_out")
IMAGE_CFG = dict(modality="image", checkpoint="flux1-schnell-fp8.safetensors", prompt="a cat",
                 seed=7, steps=4, cfg=1.0, sampler="euler", scheduler="simple",
                 width=1024, height=1024)
MUSIC_CFG = dict(modality="music", unet="minimax_music3_dit_fp16.safetensors",
                 clip="minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                 vae="minimax_music3_dav.safetensors", prompt="lo-fi beat", lyrics="x",
                 max_duration=5, seed=4242, tiled_decode=True, format="flac")

p_v, m_v = ns["build_prompt"](VIDEO_CFG)
p_i, m_i = ns["build_prompt"](IMAGE_CFG)
p_m, m_m = ns["build_prompt"](MUSIC_CFG)


def link_check(p, label):
    ids = set(p)
    bad = []
    for nid, n in p.items():
        for k, val in n["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and val[0] not in ids:
                bad.append((nid, n["class_type"], k, val))
    return check(f"{label}: all links resolve", not bad,
                 f"dangling={bad[:3]}" if bad else f"{len(p)} nodes")


def no_missing(p, label):
    MISS = object()
    bad = [k for n in p.values() for k, v in n["inputs"].items()
           if not isinstance(v, list) and not isinstance(v, (str, int, float, bool)) and v is not None]
    return check(f"{label}: no unresolved sentinels", not bad, str(bad[:3]))


# ------------------------------------------------------------------ 3. structural asserts
check("video: node count", len(p_v) == 24 and m_v["nodes"] == 24, f"len(p_v)={len(p_v)} meta={m_v['nodes']}")
check("video: turbo meta", m_v["turbo"] is True and m_v["steps"] == 4, f"{m_v.get('turbo')} steps={m_v.get('steps')}")
h3 = p_v.get("9999")
check("video: H3MultiStream splice", h3 is not None and h3["class_type"] == "H3MultiStream"
      and isinstance(h3["inputs"].get("model"), list) and h3["inputs"].get("exchange") == "host",
      json.dumps(h3["inputs"])[:120] if h3 else "node 9999 missing")
consumers = [nid for nid, n in p_v.items()
             if n["class_type"] in ("BasicGuider", "BasicScheduler") and n["inputs"].get("model") == ["9999", 0]]
check("video: H3MultiStream feeds guider+scheduler", len(consumers) == 2, str(consumers))
sv = [n for n in p_v.values() if n["class_type"] == "SaveVideo"]
check("video: SaveVideo prefix+dynamic combos", bool(sv) and sv[0]["inputs"].get("filename_prefix", "").startswith("video/")
      and isinstance(sv[0]["inputs"].get("format"), str) and sv[0]["inputs"].get("format") in ("auto", "mov", "mp4", "webm", "mkv"),
      json.dumps(sv[0]["inputs"])[:140])
sw = len([n for n in p_v.values() if n["class_type"] == "ComfySwitchNode"])
check("video: two turbo switch nodes", sw == 2, str(sw))
link_check(p_v, "video")
no_missing(p_v, "video")

check("image: node count", len(p_i) == 7 and m_i["nodes"] == 7, f"len(p_i)={len(p_i)}")
si = [n for n in p_i.values() if n["class_type"] == "SaveImage"]
check("image: SaveImage prefix", bool(si) and si[0]["inputs"].get("filename_prefix") == "h3_image", str(si[0]["inputs"]) if si else "none")
ks = [n for n in p_i.values() if n["class_type"] == "KSampler"]
check("image: KSampler 4 steps cfg 1", bool(ks) and ks[0]["inputs"].get("steps") == 4 and ks[0]["inputs"].get("cfg") == 1.0,
      json.dumps(ks[0]["inputs"])[:140] if ks else "none")
link_check(p_i, "image")
no_missing(p_i, "image")

check("music: node count", len(p_m) == 12 and m_m["nodes"] == 12, f"len(p_m)={len(p_m)}")
sa = [n for n in p_m.values() if n["class_type"] == "SaveAudioAdvanced"]
sav = sa[0]["inputs"] if sa else {}
check("music: SaveAudioAdvanced prefix", sav.get("filename_prefix") == "h3_music", str(sav))
check("music: format is the option-key STRING (v21 bug class)",
      sav.get("format") == "flac" and not isinstance(sav.get("format"), dict),
      f"format={sav.get('format')!r} type={type(sav.get('format')).__name__}")
tl = [n for n in p_m.values() if n["class_type"] == "VAEDecodeAudioTiled"]
check("music: tiled decoder wired", len(tl) == 1, str(len(tl)))
link_check(p_m, "music")
no_missing(p_m, "music")

# ------------------------------------------------------------------ 4. deep: DynamicCombo kwargs sim (needs torch + comfy_api env)
# Uses the REAL io machinery (get_finalized_class_inputs + build_nested_inputs) with a
# schema mirroring SaveAudioAdvanced, rather than importing comfy_extras (which pulls
# comfy.model_management -> torch.cuda and only loads inside a running server).
if not args.skip_deep:
    try:
        from comfy_api.latest import _io, IO

        class SimSaveAudio(IO.ComfyNode):
            @classmethod
            def define_schema(cls):
                return IO.Schema(
                    node_id="SaveAudioAdvanced",
                    inputs=[
                        IO.Audio.Input("audio"),
                        IO.String.Input("filename_prefix", default="audio/ComfyUI"),
                        IO.DynamicCombo.Input("format", options=[
                            IO.DynamicCombo.Option("flac", []),
                            IO.DynamicCombo.Option("mp3", [IO.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0")]),
                            IO.DynamicCombo.Option("opus", [IO.Combo.Input("quality", options=["64k", "128k", "192k", "256k", "320k"], default="128k")]),
                        ]),
                    ],
                    outputs=[IO.Audio.Output("audio")],
                )
            @classmethod
            def execute(cls, audio, filename_prefix: str, format: dict):
                return IO.NodeOutput(audio)

        ci = SimSaveAudio.INPUT_TYPES()

        def kw_sim(inputs):
            valid, _h, v3 = _io.get_finalized_class_inputs(ci, dict(inputs))
            return _io.build_nested_inputs(dict(inputs), v3), valid

        kw, valid = kw_sim(sav)
        got = kw.get("format")
        check("deep: format nests to {'format':'flac'} (executor kwargs)",
              got == {"format": "flac"}, json.dumps(got))
        check("deep: 'format' survives executor schema", "format" in valid["required"],
              "required=" + str(list(valid["required"])))
        old = dict(sav, format={"format": "flac"})
        _, valid_old = kw_sim(old)
        check("deep: v20 dict form is dropped (regression won't recur)",
              "format" not in valid_old["required"], "required=" + str(list(valid_old["required"])))
        mp = dict(sav, format="mp3", **{"format.quality": "V0"})
        kwm, _ = kw_sim(mp)
        check("deep: mp3 nests quality into options dict", kwm.get("format") == {"format": "mp3", "quality": "V0"},
              json.dumps(kwm.get("format")))
    except ImportError as e:
        print(f"  [--] deep checks skipped (no local ComfyUI env): {e}")
    except Exception as e:
        check("deep: kwargs sim", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ 5. live /prompt validation
if args.live:
    saa_spec = (live_info.get("SaveAudioAdvanced") or {}).get("input", {}).get("required", {}).get("format")
    check("live: upstream SaveAudioAdvanced.format is COMFY_DYNAMICCOMBO_V3",
          isinstance(saa_spec, list) and saa_spec and saa_spec[0] == "COMFY_DYNAMICCOMBO_V3",
          str(saa_spec)[:80])
    for label, p in (("video", p_v), ("image", p_i), ("music", p_m)):
        if label == "video" and not h3_pack_present:
            print("  [--] live video /prompt skipped — H3MultiStream pack not installed locally")
            continue
        req = urllib.request.Request(args.live.rstrip("/") + "/prompt",
                                     data=json.dumps({"prompt": p}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            resp = json.load(e)
        ne = resp.get("node_errors") or {}
        allowed_loaders = {"UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple", "LoraLoaderModelOnly"}
        unexpected = []
        for nid, info in ne.items():
            for er in info.get("errors", []):
                t = er.get("type", "")
                cls = info.get("class_type", "")
                if not (t == "value_not_in_list" and cls in allowed_loaders):
                    unexpected.append(f"{nid}:{cls}:{t}:{er.get('details','')}")
        check(f"live {label}: no structural/format validation errors", not unexpected,
              f"node_errors={len(ne)} unexpected={unexpected[:4]}")
        if len(ne) == 0:
            check(f"live {label}: prompt accepted", bool(resp.get("prompt_id")),
                  f"id={resp.get('prompt_id')}")
        else:
            print(f"  [--] live {label}: {len(ne)} loader value_not_in_list errors expected "
                  "(no model files locally); acceptance not asserted")

fails = [n for n, ok in RESULTS if not ok]
print("\n" + "=" * 60)
print(f"RESULT: {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL CHECKS PASS — safe to push")