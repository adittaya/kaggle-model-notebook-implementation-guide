#!/usr/bin/env python3
"""Local CLI for the MiniMax H3 Kaggle API server."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def cmd_health(args) -> int:
    r = requests.get(f"{args.base_url}/health", timeout=30)
    print(json.dumps(r.json(), indent=2))
    return 0 if r.status_code == 200 else 1


def cmd_generate(args) -> int:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "resolution": args.resolution,
        "duration_seconds": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "quality": "maximum",
    }
    if args.sampler:
        payload["sampler"] = args.sampler

    files = []
    form_files = {}
    for field, path in (
        ("image_start", args.image_start),
        ("image_end", args.image_end),
        ("video_guide", args.video_guide),
        ("audio_guide", args.audio_guide),
    ):
        if path:
            p = Path(path)
            form_files[field] = (p.name, p.open("rb"), "application/octet-stream")
            files.append(form_files[field][1])
    for ref in args.reference_image or []:
        p = Path(ref)
        fh = p.open("rb")
        files.append(fh)
        form_files.setdefault("reference_image", []).append((p.name, fh, "application/octet-stream"))

    try:
        if form_files and any(field in form_files for field in ("image_start", "image_end", "video_guide", "audio_guide")) or form_files.get("reference_image"):
            # flatten: requests wants list of tuples for repeated fields
            multipart = []
            for field, value in form_files.items():
                if isinstance(value, list):
                    for v in value:
                        multipart.append((field, v))
                else:
                    multipart.append((field, value))
            r = requests.post(
                f"{args.base_url}/v1/videos",
                headers=headers(args.api_key),
                data={"payload": json.dumps(payload)},
                files=multipart,
                timeout=120,
            )
        else:
            r = requests.post(
                f"{args.base_url}/v1/videos",
                headers={**headers(args.api_key), "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120,
            )
    finally:
        for fh in files:
            try:
                fh.close()
            except Exception:
                pass

    if r.status_code not in (200, 202):
        print(f"Error {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    job = r.json()
    print(json.dumps(job, indent=2))
    if not args.wait:
        return 0

    job_id = job["id"]
    while True:
        time.sleep(args.poll)
        s = requests.get(f"{args.base_url}/v1/videos/{job_id}", headers=headers(args.api_key), timeout=30)
        info = s.json()
        print(f"[{info.get('status')}]")
        if info.get("status") in ("succeeded", "failed", "cancel_requested"):
            break
    if info.get("status") != "succeeded":
        print(json.dumps(info, indent=2), file=sys.stderr)
        return 1

    out = Path(args.output)
    with requests.get(
        f"{args.base_url}/v1/videos/{job_id}/download", headers=headers(args.api_key), stream=True, timeout=600
    ) as dl:
        dl.raise_for_status()
        with out.open("wb") as f:
            for chunk in dl.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved: {out.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="h3_cli.py", description="MiniMax H3 API client")
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("health",):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default=env("H3_BASE_URL"))
        p.add_argument("--api-key", default=env("H3_API_KEY"))

    g = sub.add_parser("generate")
    g.add_argument("--base-url", default=env("H3_BASE_URL"))
    g.add_argument("--api-key", default=env("H3_API_KEY"))
    g.add_argument("--model", default="fl2va", choices=["fl2va", "ref2va"])
    g.add_argument("--prompt", required=True)
    g.add_argument("--resolution", default="1344x768")
    g.add_argument("--duration", type=float, default=5.0)
    g.add_argument("--steps", type=int, default=28)
    g.add_argument("--seed", type=int, default=-1)
    g.add_argument("--sampler", default=None, choices=[None, "euler", "res_multistep", "ralston_2s"])
    g.add_argument("--image-start", default=None)
    g.add_argument("--image-end", default=None)
    g.add_argument("--video-guide", default=None)
    g.add_argument("--audio-guide", default=None)
    g.add_argument("--reference-image", action="append", default=None)
    g.add_argument("--wait", action="store_true")
    g.add_argument("--poll", type=float, default=15.0)
    g.add_argument("--output", default="result.mp4")

    args = ap.parse_args()
    if not args.base_url:
        print("Set --base-url or H3_BASE_URL", file=sys.stderr)
        return 2
    if getattr(args, "api_key", "") == "" and args.command != "health":
        print("Set --api-key or H3_API_KEY", file=sys.stderr)
        return 2

    if args.command == "health":
        return cmd_health(args)
    if args.command == "generate":
        return cmd_generate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
