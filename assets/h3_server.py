#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import psutil
import torch
import uvicorn

WAN_ROOT = Path(os.environ["WAN2GP_ROOT"]).resolve()
OUTPUT_DIR = Path(os.environ.get("H3_OUTPUT_DIR", "/tmp/h3_api_outputs")).resolve()
UPLOAD_DIR = Path(os.environ.get("H3_UPLOAD_DIR", "/tmp/h3_api_uploads")).resolve()
MODEL_TYPE_FL2VA = os.environ["H3_MODEL_TYPE_FL2VA"]
MODEL_TYPE_REF2VA = os.environ.get("H3_MODEL_TYPE_REF2VA", "")
PORT = int(os.environ.get("H3_API_PORT", "8000"))
HOST = os.environ.get("H3_API_HOST", "127.0.0.1")
API_KEY = os.environ["H3_API_KEY"]
DEFAULT_RESOLUTION = os.environ.get("H3_DEFAULT_RESOLUTION", "1344x768")
DEFAULT_STEPS = int(os.environ.get("H3_DEFAULT_STEPS", "28"))
DEFAULT_DURATION = float(os.environ.get("H3_DEFAULT_DURATION", "5.0"))
DEFAULT_SEED = int(os.environ.get("H3_DEFAULT_SEED", "-1"))
STARTUP_SMOKE_TEST = os.environ.get("H3_STARTUP_SMOKE_TEST", "1") == "1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WAN_ROOT))

from shared.api import init  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if auth[7:].strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def gpu_snapshot() -> list[dict[str, Any]]:
    rows = []
    if not torch.cuda.is_available():
        return rows
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        rows.append({
            "index": idx,
            "name": props.name,
            "total_vram_gb": round(total_b / 2**30, 2),
            "free_vram_gb": round(free_b / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return rows


def normalize_resolution(value: str | None) -> str:
    value = str(value or DEFAULT_RESOLUTION).strip().lower().replace(" ", "")
    if "x" not in value:
        raise ValueError("resolution must be WIDTHxHEIGHT")
    w, h = value.split("x", 1)
    try:
        wi, hi = int(w), int(h)
    except ValueError as exc:
        raise ValueError("resolution must contain integer dimensions") from exc
    if wi < 256 or hi < 256:
        raise ValueError("resolution is too small")
    if wi % 32 or hi % 32:
        raise ValueError("width and height must be multiples of 32")
    if wi * hi > 2_200_000:
        raise ValueError("resolution exceeds the conservative 2.2 MP Kaggle safety limit")
    return f"{wi}x{hi}"


def set_if_present(settings: dict[str, Any], key: str, value: Any) -> None:
    if key in settings:
        settings[key] = value


@dataclass
class JobRecord:
    id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    request: dict[str, Any] | None = None
    model_type: str | None = None
    generated_files: list[str] | None = None
    primary_file: str | None = None
    error: str | None = None


SESSION = None
JOB_LOCK = threading.Lock()
JOBS: dict[str, JobRecord] = {}
ACTIVE_JOB_ID: str | None = None
READY = False
STARTUP_ERROR: str | None = None
SMOKE_OUTPUT: str | None = None


def model_type_from_public_name(name: str | None) -> str:
    n = str(name or "fl2va").strip().lower()
    if n in {"fl2va", "h3", "h3-fl2va"}:
        return MODEL_TYPE_FL2VA
    if n in {"ref2va", "reference", "h3-ref2va"}:
        if not MODEL_TYPE_REF2VA:
            raise ValueError("Ref2VA is not enabled")
        return MODEL_TYPE_REF2VA
    raise ValueError("model must be fl2va or ref2va")


def pick_primary(files: list[str]) -> str | None:
    for p in files:
        if str(p).lower().endswith((".mp4", ".mov", ".webm", ".mkv")) and Path(p).exists():
            return str(Path(p).resolve())
    for p in files:
        if Path(p).exists():
            return str(Path(p).resolve())
    return None


def build_settings(payload: dict[str, Any], media: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_type = model_type_from_public_name(payload.get("model", "fl2va"))
    settings = SESSION.get_default_settings(model_type)
    settings["model_type"] = model_type

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    settings["prompt"] = prompt

    set_if_present(settings, "resolution", normalize_resolution(payload.get("resolution")))
    set_if_present(settings, "duration_seconds", float(payload.get("duration_seconds", DEFAULT_DURATION)))
    set_if_present(settings, "num_inference_steps", int(payload.get("steps", DEFAULT_STEPS)))
    set_if_present(settings, "seed", int(payload.get("seed", DEFAULT_SEED)))

    quality = str(payload.get("quality", "maximum")).lower()
    if quality not in {"maximum", "max", "quality"}:
        raise ValueError("This quality-first server only exposes quality='maximum'.")

    sampler = payload.get("sampler")
    if sampler:
        sampler = str(sampler).lower()
        if sampler not in {"euler", "res_multistep", "ralston_2s"}:
            raise ValueError("sampler must be euler, res_multistep, or ralston_2s")
        for key in ("sample_solver", "sampler", "scheduler"):
            set_if_present(settings, key, sampler)

    for key in (
        "image_start", "image_end", "image_refs",
        "video_guide", "video_guide2", "video_guide3",
        "audio_guide", "audio_guide2", "audio_guide3",
    ):
        if media.get(key):
            settings[key] = media[key]

    return model_type, settings


async def save_upload(upload: UploadFile, prefix: str) -> str:
    name = Path(upload.filename or "upload.bin").name
    dest = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}_{name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest.resolve())


async def parse_input(request: Request) -> tuple[dict[str, Any], dict[str, Any]]:
    content_type = request.headers.get("content-type", "").lower()
    media: dict[str, Any] = {}
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        raw = form.get("payload") or form.get("request") or "{}"
        payload = json.loads(str(raw))

        mapping = {
            "image_start": "image_start",
            "image_end": "image_end",
            "video_guide": "video_guide",
            "video_guide2": "video_guide2",
            "video_guide3": "video_guide3",
            "audio_guide": "audio_guide",
            "audio_guide2": "audio_guide2",
            "audio_guide3": "audio_guide3",
        }
        for field, out_name in mapping.items():
            item = form.get(field)
            if isinstance(item, UploadFile):
                media[out_name] = await save_upload(item, field)

        refs = []
        getlist = getattr(form, "getlist", None)
        if callable(getlist):
            for item in getlist("reference_image"):
                if isinstance(item, UploadFile):
                    refs.append(await save_upload(item, "ref"))
        elif isinstance(form.get("reference_image"), UploadFile):
            refs.append(await save_upload(form["reference_image"], "ref"))
        if refs:
            media["image_refs"] = refs
        return payload, media

    body = await request.body()
    payload = json.loads(body.decode("utf-8") or "{}")
    return payload, media


def run_job(job_id: str, settings: dict[str, Any]) -> None:
    global ACTIVE_JOB_ID
    record = JOBS[job_id]
    record.status = "running"
    record.started_at = utc_now()
    try:
        job = SESSION.submit_task(settings)
        result = job.result(timeout=None)
        files = [str(Path(p).resolve()) for p in (result.generated_files or []) if Path(p).exists()]
        record.generated_files = files
        record.primary_file = pick_primary(files)
        record.status = "succeeded" if result.success else "failed"
        record.error = None if result.success else (
            "\n".join(getattr(e, "message", str(e)) for e in (result.errors or [])) or "Generation failed"
        )
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
    finally:
        record.finished_at = utc_now()
        with JOB_LOCK:
            ACTIVE_JOB_ID = None


def startup_smoke() -> None:
    global SMOKE_OUTPUT
    if not STARTUP_SMOKE_TEST:
        return
    settings = SESSION.get_default_settings(MODEL_TYPE_FL2VA)
    settings["model_type"] = MODEL_TYPE_FL2VA
    settings["prompt"] = (
        "integrated_multimodal_description: [Shot 1] A cinematic single take in a "
        "rainy glass observatory at night. A scientist turns toward a glowing console "
        "with a small natural smile; subtle slow camera push; realistic materials, "
        "stable anatomy and physically coherent motion. overall_soundscape: rain on "
        "glass, low electrical hum, one clean relay click, quiet room tone."
    )
    set_if_present(settings, "resolution", DEFAULT_RESOLUTION)
    set_if_present(settings, "duration_seconds", 4.5)
    set_if_present(settings, "num_inference_steps", 8)
    set_if_present(settings, "seed", 123456)
    result = SESSION.submit_task(settings).result(timeout=None)
    if not result.success:
        errs = [f"stage={getattr(e, 'stage', None)}: {getattr(e, 'message', str(e))}" for e in (result.errors or [])]
        for line in errs:
            print("SMOKE ERROR:", line, flush=True)
        raise RuntimeError("\n".join(errs) or "Startup smoke test failed")
    files = [str(Path(p).resolve()) for p in (result.generated_files or []) if Path(p).exists()]
    SMOKE_OUTPUT = pick_primary(files)
    if not SMOKE_OUTPUT:
        raise RuntimeError("Startup smoke test produced no media file")


def initialize() -> None:
    global SESSION, READY, STARTUP_ERROR
    try:
        SESSION = init(
            root=str(WAN_ROOT),
            output_dir=str(OUTPUT_DIR),
            cli_args=(
                "--profile", "4",
                "--attention", "sdpa",
                "--vram-allocator", "vmm_spill",
                "--perc-reserved-mem-max", "0.45",
                "--verbose", "1",
            ),
            console_output=True,
            console_isatty=False,
        )
        for model_type in (MODEL_TYPE_FL2VA, MODEL_TYPE_REF2VA):
            if model_type and SESSION.get_model_def(model_type) is None:
                raise RuntimeError(f"Model definition not found: {model_type}")
        startup_smoke()
        READY = True
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        READY = False
        raise


app = FastAPI(title="MiniMax H3 Kaggle API", version="1.0.0", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> JSONResponse:
    body = {
        "status": "ready" if READY else "starting" if STARTUP_ERROR is None else "failed",
        "ready": READY,
        "model": "MiniMax H3",
        "models": {"fl2va": MODEL_TYPE_FL2VA, "ref2va": MODEL_TYPE_REF2VA or None},
        "quality_defaults": {
            "resolution": DEFAULT_RESOLUTION,
            "duration_seconds": DEFAULT_DURATION,
            "steps": DEFAULT_STEPS,
            "turbo": False,
            "step_caches": False,
            "attention": "sdpa",
            "profile": 4,
            "quality": "maximum",
        },
        "gpu": gpu_snapshot(),
        "smoke_test_output": SMOKE_OUTPUT,
        "error": STARTUP_ERROR,
    }
    return JSONResponse(body, status_code=200 if READY else 503 if STARTUP_ERROR else 200)


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    require_auth(request)
    data = []
    for public_name, model_type in (("fl2va", MODEL_TYPE_FL2VA), ("ref2va", MODEL_TYPE_REF2VA)):
        if not model_type:
            continue
        metadata = SESSION.get_model_metadata(model_type, include_availability=True) or {}
        data.append({
            "id": public_name,
            "model_type": model_type,
            "name": metadata.get("name", model_type),
            "description": metadata.get("description", ""),
            "availability": metadata.get("availability"),
        })
    return JSONResponse({"data": data})


@app.get("/v1/system")
async def system_info(request: Request) -> JSONResponse:
    require_auth(request)
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(OUTPUT_DIR)
    return JSONResponse({
        "gpu": gpu_snapshot(),
        "ram": {
            "total_gb": round(vm.total / 2**30, 2),
            "available_gb": round(vm.available / 2**30, 2),
        },
        "output_free_gb": round(disk.free / 2**30, 2),
        "active_job": ACTIVE_JOB_ID,
    })


@app.post("/v1/videos")
async def create_video(request: Request) -> JSONResponse:
    global ACTIVE_JOB_ID
    require_auth(request)
    if not READY:
        raise HTTPException(status_code=503, detail="H3 server is not ready")

    try:
        payload, media = await parse_input(request)
        model_type, settings = build_settings(payload, media)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {type(exc).__name__}: {exc}") from exc

    with JOB_LOCK:
        if ACTIVE_JOB_ID:
            active = JOBS.get(ACTIVE_JOB_ID)
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A quality-first H3 generation is already running.",
                    "active_job_id": ACTIVE_JOB_ID,
                    "active_status": active.status if active else "running",
                },
            )
        job_id = f"h3_{uuid.uuid4().hex[:16]}"
        JOBS[job_id] = JobRecord(
            id=job_id,
            status="queued",
            created_at=utc_now(),
            request=payload,
            model_type=model_type,
        )
        ACTIVE_JOB_ID = job_id

    threading.Thread(target=run_job, args=(job_id, settings), daemon=True).start()
    return JSONResponse({
        "id": job_id,
        "status": "queued",
        "model": payload.get("model", "fl2va"),
        "model_type": model_type,
        "created_at": JOBS[job_id].created_at,
    }, status_code=202)


@app.get("/v1/videos/{job_id}")
async def get_video(job_id: str, request: Request) -> JSONResponse:
    require_auth(request)
    record = JOBS.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JSONResponse({
        "id": record.id,
        "status": record.status,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "model_type": record.model_type,
        "download_url": f"/v1/videos/{record.id}/download" if record.primary_file else None,
        "generated_files": [Path(x).name for x in (record.generated_files or [])],
        "error": record.error,
    })


@app.get("/v1/videos/{job_id}/download")
async def download_video(job_id: str, request: Request) -> FileResponse:
    require_auth(request)
    record = JOBS.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if record.status != "succeeded" or not record.primary_file:
        raise HTTPException(status_code=409, detail="Job is not completed successfully")
    path = Path(record.primary_file)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Generated file is no longer available")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.post("/v1/videos/{job_id}/cancel")
async def cancel_video(job_id: str, request: Request) -> JSONResponse:
    require_auth(request)
    record = JOBS.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown job id")
    active = getattr(SESSION, "_active_job", None)
    if active is not None:
        try:
            active.cancel()
        except Exception:
            pass
    record.status = "cancel_requested"
    return JSONResponse({"id": job_id, "status": record.status})


if __name__ == "__main__":
    initialize()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
