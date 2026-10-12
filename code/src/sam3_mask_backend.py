from __future__ import annotations

import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"
DEFAULT_SAM3_CKPT = PROJECT_ROOT / "weights" / "sam3" / "sam3.pt"
DEFAULT_SAM3_PROMPTS = ["object", "robot arm", "moving object"]
MIN_MASK_BYTES = 1024


def _split_prompts(value) -> list[str]:
    if value is None:
        return list(DEFAULT_SAM3_PROMPTS)
    if isinstance(value, str):
        prompts = [part.strip() for part in value.split("|")]
    else:
        prompts = [str(part).strip() for part in value]
    prompts = [prompt for prompt in prompts if prompt]
    return prompts or list(DEFAULT_SAM3_PROMPTS)


def _resolve_python(python_bin: str | None) -> str:
    if python_bin:
        return str(Path(python_bin).expanduser())
    return sys.executable


def _device_to_gpu_token(device: str | None) -> str:
    if not device:
        return "0"
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    if value.startswith("gpu") and value[3:].isdigit():
        return value[3:]
    if value == "cuda":
        return "0"
    return "0"


def _video_cache_name(video_path: Path) -> str:
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:12]
    return f"{video_path.stem}_{digest}_sam3_front_mask.mp4"


def _default_out_path(opts, video_path: Path) -> Path:
    out_dir_arg = getattr(opts, "sam3_output_dir", None)
    if out_dir_arg:
        out_dir = Path(out_dir_arg).expanduser().resolve()
    else:
        # Batch sweeps run left/right rotations of the same input. Keep this
        # cache at run scope so those rotations share a mask generated from the
        # input video, without ever reading CSV-provided front_mask paths.
        run_dir = Path(getattr(opts, "out_dir", None) or Path(opts.save_dir).parent)
        out_dir = run_dir.expanduser().resolve() / "_sam3_front_mask_cache"
    return out_dir / _video_cache_name(video_path)


def _mask_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > MIN_MASK_BYTES


def _wait_for_cached_mask(path: Path, lock_dir: Path, max_wait_s: int = 3600) -> bool:
    start = time.time()
    while time.time() - start < max_wait_s:
        if _mask_ready(path):
            print(f"[sam3] reusing cached front mask: {path}", flush=True)
            return True
        if not lock_dir.exists():
            return False
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for SAM3 mask cache lock: {lock_dir}")


def generate_sam3_front_mask(opts) -> str:
    video_path = Path(opts.video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"SAM3 input video not found: {video_path}")

    out_path = _default_out_path(opts, video_path)
    force_mask = getattr(opts, "sam3_force_mask", False)
    can_reuse = (
        not force_mask
        and (getattr(opts, "sam3_reuse_mask", False) or not getattr(opts, "sam3_output_dir", None))
    )
    if can_reuse and _mask_ready(out_path):
        print(f"[sam3] reusing cached front mask: {out_path}", flush=True)
        return str(out_path)

    helper_path = Path(__file__).resolve().parent / "sam3_generate_mask.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"SAM3 helper script not found: {helper_path}")

    prompts = _split_prompts(getattr(opts, "sam3_prompts", None))
    python_bin = _resolve_python(getattr(opts, "sam3_python_bin", None))
    sam3_repo = str(
        Path(getattr(opts, "sam3_repo_path", None) or DEFAULT_SAM3_REPO)
        .expanduser()
        .resolve()
    )
    checkpoint_path = str(
        Path(getattr(opts, "sam3_checkpoint_path", None) or DEFAULT_SAM3_CKPT)
        .expanduser()
        .resolve()
    )
    gpus = str(
        getattr(opts, "sam3_gpus", None)
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or _device_to_gpu_token(getattr(opts, "device", None))
    )

    cmd = [
        python_bin,
        str(helper_path),
        "--video_path",
        str(video_path),
        "--out_path",
        str(out_path),
        "--sam3_repo_path",
        sam3_repo,
        "--checkpoint_path",
        checkpoint_path,
        "--prompt_frame_index",
        str(int(getattr(opts, "sam3_prompt_frame_index", 0) or 0)),
        "--gpus",
        gpus,
        "--prompts",
        *prompts,
    ]
    if force_mask:
        cmd.append("--force")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = out_path.with_name(out_path.name + ".lock")
    acquired_lock = False
    while not acquired_lock:
        if can_reuse and _wait_for_cached_mask(out_path, lock_dir):
            return str(out_path)
        try:
            lock_dir.mkdir()
            acquired_lock = True
        except FileExistsError:
            time.sleep(2)
            continue

    try:
        if can_reuse and _mask_ready(out_path):
            print(f"[sam3] reusing cached front mask: {out_path}", flush=True)
            return str(out_path)
        print("[sam3] generating front mask: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        if not _mask_ready(out_path):
            raise FileNotFoundError(f"SAM3 did not create expected mask: {out_path}")
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass
    return str(out_path)
