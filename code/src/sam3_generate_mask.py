#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"
DEFAULT_SAM3_CKPT = PROJECT_ROOT / "weights" / "sam3" / "sam3.pt"


def _add_sam3_repo_to_path(sam3_repo: Path) -> None:
    repo_str = str(sam3_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_sam3_demo(sam3_repo: Path):
    demo_path = sam3_repo / "demo.py"
    if not demo_path.is_file():
        raise FileNotFoundError(f"SAM3 demo.py not found: {demo_path}")
    _add_sam3_repo_to_path(sam3_repo)
    spec = importlib.util.spec_from_file_location("embody4d_sam3_demo", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import SAM3 demo from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_video(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SAM3 output mask not found: {path}")
    if path.stat().st_size < 1024:
        raise RuntimeError(f"SAM3 output mask is too small: {path} ({path.stat().st_size} bytes)")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe could not read SAM3 mask: {path}\n{result.stderr[:500]}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one front mask video with SAM3.")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--sam3_repo_path", default=str(DEFAULT_SAM3_REPO))
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--prompts", nargs="+", default=["object", "robot arm", "moving object"])
    parser.add_argument("--prompt_frame_index", type=int, default=0)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    sam3_repo = Path(args.sam3_repo_path).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not sam3_repo.is_dir():
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_repo}")
    checkpoint_path = (
        Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else DEFAULT_SAM3_CKPT
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Local SAM3 checkpoint not found: {checkpoint_path}. "
            "Set --checkpoint_path to a valid local checkpoint."
        )
    if out_path.is_file() and out_path.stat().st_size > 1024 and not args.force:
        print(f"[sam3] reuse existing mask: {out_path}", flush=True)
        _verify_video(out_path)
        print(str(out_path), flush=True)
        return 0

    visible = str(args.gpus).strip()
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _add_sam3_repo_to_path(sam3_repo)
    from sam3.model_builder import build_sam3_video_predictor

    demo = _load_sam3_demo(sam3_repo)
    predictor_kwargs = {
        "gpus_to_use": [0],
        "checkpoint_path": str(checkpoint_path),
    }
    predictor = build_sam3_video_predictor(**predictor_kwargs)
    try:
        print(
            "[sam3] processing "
            f"{video_path} -> {out_path} prompts={args.prompts}",
            flush=True,
        )
        result = demo.process_one_video(
            predictor=predictor,
            video_path=str(video_path),
            out_dir=str(out_path.parent),
            prompt_texts=list(args.prompts),
            prompt_frame_index=int(args.prompt_frame_index),
            out_path_override=str(out_path),
        )
    finally:
        predictor.shutdown()

    result_path = Path(result)
    _verify_video(result_path)
    print(str(result_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
