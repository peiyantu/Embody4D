"""
Batch video mask generation with SAM 3.

Based on `examples/sam3_video_predictor_example.ipynb`.

This script:
- traverses all video files under an input directory
- runs SAM 3 video predictor with text prompts
- writes a mask video per input video under an output directory
  (same basename + "_mask" before extension)
"""

from __future__ import annotations

import argparse
import os
import multiprocessing
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def list_videos(root: str, exclude_dirs: Sequence[str] | None = None) -> List[str]:
    paths: List[str] = []
    root_abs = os.path.abspath(root)
    exclude_abs = [os.path.abspath(d) for d in (exclude_dirs or [])]

    def _is_under(p: str, parent: str) -> bool:
        try:
            return os.path.commonpath([p, parent]) == parent
        except ValueError:
            return False

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_abs = os.path.abspath(dirpath)
        if any(_is_under(dirpath_abs, ex) for ex in exclude_abs):
            dirnames[:] = []
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def filter_readable_videos(
    video_paths: List[str],
) -> Tuple[List[str], List[str]]:
    """Split paths into readable vs permission-denied (or missing)."""
    readable: List[str] = []
    skipped: List[str] = []
    for p in video_paths:
        if not os.path.exists(p):
            skipped.append(p)
            continue
        if not os.access(p, os.R_OK):
            skipped.append(p)
            continue
        readable.append(p)
    return readable, skipped


def parse_errors_log(log_path: str) -> List[str]:
    """
    Parse fast_scan_errors.log; return unique mask file paths (CORRUPT or MISSING).
    Line format: Sample N | Key: ... | CORRUPT|MISSING | /path/to/mask.mp4 [| Error: ...]
    """
    seen: set = set()
    out: List[str] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "| CORRUPT | " in line:
                part = line.split("| CORRUPT | ", 1)[-1]
            elif "| MISSING | " in line:
                part = line.split("| MISSING | ", 1)[-1]
            else:
                continue
            path = part.split(" | ")[0].strip()
            if path and path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _mirror_out_path(
    video_path: str,
    in_dir: str,
    out_dir: str,
    anchor_dir_name: str | None = None,
) -> str:
    """
    Compute output path that mirrors the input directory structure.

    Example:
        video_path = /data/video/A/B/clip.mp4
        in_dir     = /data/video
        out_dir    = /data/mask_128
        -> /data/mask_128/A/B/clip.mp4
    """
    vp_abs = os.path.abspath(video_path)
    in_abs = os.path.abspath(in_dir)

    # 1) Prefer anchoring by a stable directory name (keeps structure even if --in_dir points
    #    to an episode folder or you are regenerating from an errors log).
    if anchor_dir_name:
        marker = f"/{anchor_dir_name}/"
        if marker in vp_abs:
            rel = vp_abs.split(marker, 1)[1]
            return os.path.join(out_dir, rel)

    # 2) Normal case: vp is under in_dir.
    try:
        common = os.path.commonpath([vp_abs, in_abs])
    except ValueError:
        common = ""
    if common == in_abs:
        rel = os.path.relpath(vp_abs, in_abs)
        # If the user points --in_dir at an episode folder (e.g. .../episode_000000),
        # keep that directory level in the output to avoid flattening.
        if os.path.dirname(rel) == "":
            base_in = os.path.basename(in_abs)
            if base_in.startswith("episode_"):
                rel = os.path.join(base_in, rel)
        return os.path.join(out_dir, rel)

    # 3) Fallback: preserve full absolute structure (collision-free; avoids dropping `episode_*/`).
    return os.path.join(out_dir, vp_abs.lstrip("/"))


def mask_path_to_source_path(mask_path: str) -> str:
    """
    Derive source video path from a mask path in fast_scan_errors.log.
    - Filename: episode_000021mask.mp4 -> episode_000021.mp4
    - Dir: .../mask_object/... -> .../... (remove mask_object segment)
    - Dir: .../mask_128/... -> .../video/... (mask_128 -> video for dataset-specific)
    """
    d = os.path.dirname(mask_path)
    bn = os.path.basename(mask_path)
    if bn.endswith("mask.mp4"):
        base = bn[:-8]  # strip 'mask.mp4' (8 chars)
        ext = ".mp4"
    else:
        base, ext = os.path.splitext(bn)
        if base.endswith("mask"):
            base = base[:-4]
    name = base + ext
    if "/mask_object/" in d:
        d = d.replace("/mask_object/", "/", 1)
    # mask_128 may be path segment or full dir name (with or without trailing slash)
    if "/mask_128/" in d:
        d = d.replace("/mask_128/", "/video/", 1)
    elif d.endswith("/mask_128"):
        d = d[:-9] + "/video"
    return os.path.join(d, name)


def get_video_meta(video_path: str) -> Tuple[int, int, float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size for {video_path}: {width}x{height}")
    if fps <= 0:
        fps = 10.0
    return width, height, fps


def _to_bool_mask(mask):
    import numpy as np

    # expected shapes:
    # - (H, W) with {0,1} / {0,255} / logits-like
    # - bool mask
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype == np.uint8:
        return mask > 0
    return mask > 0.0


def write_mask_video_mp4(
    out_path: str, frames_iter: Iterable[np.ndarray], width: int, height: int, fps: float
) -> None:
    import cv2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Materialize so we can verify frame count and avoid empty/corrupt output
    frames_list = list(frames_iter)
    n_frames = len(frames_list)
    if n_frames == 0:
        raise RuntimeError("Cannot write mask video: no frames produced (empty propagation output)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {tmp_path}")
    try:
        for frame_bgr in frames_list:
            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
            writer.write(frame_bgr)
    finally:
        writer.release()

    tmp_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    if tmp_size < 1024:
        raise RuntimeError(
            f"VideoWriter produced a tiny file ({tmp_size} bytes, {n_frames} frames); "
            "likely OpenCV/backend issue"
        )

    # Re-encode to H.264/yuv420p for wide compatibility; fallback to raw mp4v if ffmpeg fails.
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                out_path,
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except subprocess.CalledProcessError as e:
        # ffmpeg failed; keep raw mp4v so we don't leave a corrupt/old file
        if os.path.exists(tmp_path):
            try:
                os.replace(tmp_path, out_path)
            except OSError:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.replace(tmp_path, out_path)
            except OSError:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


@dataclass(frozen=True)
class PromptGroup:
    text: str
    value: int  # 0-255 grayscale value for this group in output mask


def propagate_outputs_per_frame(
    predictor, session_id: str, propagation_direction: str = "both"
) -> Dict[int, Dict]:
    outputs_per_frame: Dict[int, Dict] = {}
    for resp in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=propagation_direction,
        )
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]
    return outputs_per_frame


def _merge_outputs_per_frame(
    outputs_list: List[Dict[int, Dict]], width: int, height: int
) -> Dict[int, Dict]:
    """Union masks from multiple runs (one per prompt) into one outputs_per_frame."""
    import numpy as np
    import torch
    import cv2

    merged: Dict[int, np.ndarray] = {}
    for outputs_per_frame in outputs_list:
        for frame_idx in outputs_per_frame:
            out = outputs_per_frame[frame_idx]
            oids = out.get("out_obj_ids")
            masks = out.get("out_binary_masks")
            if frame_idx not in merged:
                merged[frame_idx] = np.zeros((height, width), dtype=np.uint8)
            if oids is None or masks is None:
                continue
            if hasattr(masks, "shape") and masks.size == 0:
                continue
            masks_np = (
                np.asarray(masks)
                if not torch.is_tensor(masks)
                else masks.detach().cpu().numpy()
            )
            for i in range(masks_np.shape[0]):
                m = _to_bool_mask(masks_np[i])
                if m.shape != (height, width):
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                merged[frame_idx][m] = 255
    return {
        fi: {
            "out_obj_ids": np.array([0]),
            "out_binary_masks": (gray[None, ...] > 0).astype(np.bool_),
        }
        for fi, gray in merged.items()
    }


def outputs_to_mask_frames(
    outputs_per_frame: Dict[int, Dict],
    width: int,
    height: int,
) -> Iterable[np.ndarray]:
    import cv2
    import numpy as np
    import torch

    """
    Convert predictor outputs to BGR frames for VideoWriter.

    Output encoding:
    - background: 0
    - any predicted object mask: 255
    """
    for frame_idx in sorted(outputs_per_frame.keys()):
        out = outputs_per_frame[frame_idx]
        obj_ids = out.get("out_obj_ids", None)
        masks = out.get("out_binary_masks", None)
        if obj_ids is None or masks is None:
            # empty frame
            gray = np.zeros((height, width), dtype=np.uint8)
        else:
            if torch.is_tensor(masks):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks)

            gray = np.zeros((masks_np.shape[1], masks_np.shape[2]), dtype=np.uint8)
            for i in range(masks_np.shape[0]):
                m = _to_bool_mask(masks_np[i])
                gray[m] = 255

        if gray.shape != (height, width):
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        yield bgr


def process_one_video(
    predictor,
    video_path: str,
    out_dir: str,
    prompt_texts: List[str],
    prompt_frame_index: int = 0,
    out_path_override: str | None = None,
) -> str:
    import torch

    out_path = out_path_override if out_path_override is not None else os.path.join(
        out_dir, os.path.basename(video_path)
    )

    width, height, fps = get_video_meta(video_path)

    resp = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = resp["session_id"]

    try:
        outputs_list: List[Dict[int, Dict]] = []

        for prompt_text in prompt_texts:
            _ = predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )
            add_resp = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_index,
                    text=prompt_text,
                )
            )
            outputs_per_frame: Dict[int, Dict] = {
                add_resp["frame_index"]: add_resp["outputs"],
            }
            for resp in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="both",
                )
            ):
                outputs_per_frame[resp["frame_index"]] = resp["outputs"]
            outputs_list.append(outputs_per_frame)

        if len(outputs_list) == 1:
            merged_outputs = outputs_list[0]
        else:
            merged_outputs = _merge_outputs_per_frame(
                outputs_list, width, height
            )

        frames_iter = outputs_to_mask_frames(
            outputs_per_frame=merged_outputs,
            width=width,
            height=height,
        )
        write_mask_video_mp4(
            out_path, frames_iter, width=width, height=height, fps=fps
        )
        return out_path
    finally:
        _ = predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )


def _get_visible_cuda_tokens(num_visible_gpus: int) -> List[str]:
    """
    Return CUDA_VISIBLE_DEVICES tokens.

    - If CUDA_VISIBLE_DEVICES is set (e.g. "0,2,3"), return ["0","2","3"].
    - Otherwise return ["0","1",...].
    """
    s = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if s:
        toks = [t.strip() for t in s.split(",") if t.strip()]
        # If user set something odd, fall back to local indexing.
        if len(toks) >= num_visible_gpus:
            return toks[:num_visible_gpus]
    return [str(i) for i in range(num_visible_gpus)]


def worker_run(
    gpu_token: str,
    gpu_label: str,
    video_tasks: List[Tuple[str, Optional[str]]],
    out_dir: str,
    prompt_texts: List[str],
    prompt_frame_index: int,
    force: bool,
    out_q,
) -> None:
    """
    Top-level worker entry (must be picklable for multiprocessing spawn).
    video_tasks: list of (video_path, out_path_override). out_path_override is None to use out_dir + basename.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token

    # IMPORTANT:
    # Delay importing `sam3.model_builder` until AFTER we set CUDA_VISIBLE_DEVICES.
    # `sam3.model_builder` runs CUDA queries at import time (e.g. get_device_properties(0)),
    # and importing it too early would pin all workers to the first visible GPU.
    from sam3.model_builder import build_sam3_video_predictor  # local import by design
    import torch

    # a few speed-friendly defaults (may increase VRAM use a bit)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # CUDA will see exactly one GPU in this process
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    try:
        for vp, out_path_override in video_tasks:
            try:
                if (not force) and out_path_override and os.path.exists(out_path_override):
                    out_q.put(("skip", vp, out_path_override, gpu_label))
                    continue
                use_dir = os.path.dirname(out_path_override) if out_path_override else out_dir
                out_path = process_one_video(
                    predictor=predictor,
                    video_path=vp,
                    out_dir=use_dir,
                    prompt_texts=prompt_texts,
                    prompt_frame_index=prompt_frame_index,
                    out_path_override=out_path_override,
                )
                out_q.put(("ok", vp, out_path, gpu_label))
            except Exception as e:
                out_q.put(("err", vp, repr(e), gpu_label))
    finally:
        predictor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        default=os.path.join(PROJECT_ROOT, "assets", "example"),
        help="Input directory containing videos (recursively searched).",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "sam3_demo"),
        help="Output directory for mask videos.",
    )
    parser.add_argument(
        "--rel_anchor",
        default=None,
        help=(
            "If set, preserve the path suffix after this directory name when writing outputs. "
            "Example: rel_anchor=observation.images.image_2 will map "
            ".../observation.images.image_2/episode_000000/seg_10.mp4 -> "
            "<out_dir>/episode_000000/seg_10.mp4"
        ),
    )
    parser.add_argument(
        "--exclude_dirs",
        nargs="*",
        default=None,
        help="Optional directory paths to exclude when scanning --in_dir (useful if out_dir is inside in_dir).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print input->output mappings (first 50) and exit without running SAM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (disable skip-on-exists).",
    )
    parser.add_argument(
        "--prompt_frame_index",
        type=int,
        default=0,
        help="Frame index to add text prompts on.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        # default=["object", "robot arm", "moving object"],
        default=["moving object"],
        help="Text prompts. Each prompt runs separately; masks are merged (union).",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="How many parallel worker processes per GPU (used when --num_workers not set).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Total parallel workers (batches). If set, overrides workers_per_gpu. E.g. 5 with --gpus 0 1 => 5 workers on GPU 0,1 round-robin.",
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="GPU ids to use (e.g. 0 1). If not set, use all visible GPUs.",
    )
    parser.add_argument(
        "--errors_log",
        default=None,
        metavar="PATH",
        help="Path to fast_scan_errors.log. If set, only regenerate mask videos for CORRUPT/MISSING paths in the log (source path derived from mask path).",
    )
    args = parser.parse_args()

    if args.errors_log:
        if not os.path.isfile(args.errors_log):
            raise RuntimeError(f"Errors log not found: {args.errors_log}")
        mask_paths = parse_errors_log(args.errors_log)
        video_tasks: List[Tuple[str, Optional[str]]] = []
        missing_sources: List[str] = []
        for mask_path in mask_paths:
            src = mask_path_to_source_path(mask_path)
            if os.path.isfile(src):
                out_path = _mirror_out_path(
                    src,
                    args.in_dir,
                    args.out_dir,
                    anchor_dir_name=str(args.rel_anchor) if args.rel_anchor else None,
                )
                video_tasks.append((src, out_path))
            else:
                missing_sources.append(src)
        if missing_sources and len(missing_sources) <= 10:
            for s in missing_sources:
                print(f"[Skip] Source not found: {s}")
        elif missing_sources:
            print(f"[Skip] {len(missing_sources)} mask entries have missing source video (first 5: {missing_sources[:5]})")
        if not video_tasks:
            raise RuntimeError("No source videos found for any entry in the errors log.")
        print(f"[Errors-log] Regenerating {len(video_tasks)} mask videos from {args.errors_log}")
    else:
        exclude_dirs: List[str] = []
        if args.exclude_dirs:
            exclude_dirs.extend([str(p) for p in args.exclude_dirs])
        # If out_dir is inside in_dir, automatically exclude it to avoid re-processing outputs.
        try:
            if os.path.commonpath([os.path.abspath(args.in_dir), os.path.abspath(args.out_dir)]) == os.path.abspath(
                args.in_dir
            ):
                exclude_dirs.append(str(args.out_dir))
        except ValueError:
            pass

        videos = list_videos(args.in_dir, exclude_dirs=exclude_dirs)
        if not videos:
            raise RuntimeError(f"No videos found under: {args.in_dir}")
        video_tasks = [
            (
                v,
                _mirror_out_path(
                    v,
                    args.in_dir,
                    args.out_dir,
                    anchor_dir_name=str(args.rel_anchor) if args.rel_anchor else None,
                ),
            )
            for v in videos
        ]

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dry_run:
        print(f"[Dry-run] tasks={len(video_tasks)} (showing up to 50)")
        for vp, op in video_tasks[:50]:
            print(f"{vp} -> {op}")
        return

    import torch

    if torch.cuda.device_count() <= 0:
        raise RuntimeError("SAM3 video predictor requires CUDA GPUs, but none were found.")

    prompt_list = [p.strip() for p in args.prompts if p.strip()]
    if not prompt_list:
        raise RuntimeError("Empty --prompts after stripping.")

    if args.gpus is not None and len(args.gpus) > 0:
        visible_gpu_tokens = [str(g) for g in args.gpus]
    else:
        num_gpus = torch.cuda.device_count()
        visible_gpu_tokens = _get_visible_cuda_tokens(num_gpus)

    if args.num_workers is not None and args.num_workers > 0:
        total_workers = int(args.num_workers)
    else:
        workers_per_gpu = max(1, int(args.workers_per_gpu))
        total_workers = len(visible_gpu_tokens) * workers_per_gpu

    def chunk_round_robin(
        items: List[Tuple[str, Optional[str]]], k: int
    ) -> List[List[Tuple[str, Optional[str]]]]:
        chunks: List[List[Tuple[str, Optional[str]]]] = [[] for _ in range(k)]
        for i, it in enumerate(items):
            chunks[i % k].append(it)
        return chunks

    # Split tasks into total_workers batches; each worker pinned to one GPU (round-robin).
    task_splits = chunk_round_robin(video_tasks, total_workers)
    worker_gpu_labels = [
        visible_gpu_tokens[i % len(visible_gpu_tokens)] for i in range(total_workers)
    ]

    ctx = multiprocessing.get_context("spawn")
    procs: List[multiprocessing.Process] = []
    q: multiprocessing.Queue = ctx.Queue()

    for worker_idx in range(total_workers):
        tasks = task_splits[worker_idx]
        if not tasks:
            continue
        gpu_token = visible_gpu_tokens[worker_idx % len(visible_gpu_tokens)]
        gpu_label = worker_gpu_labels[worker_idx]
        p = ctx.Process(
            target=worker_run,
            args=(
                gpu_token,
                gpu_label,
                tasks,
                args.out_dir,
                prompt_list,
                args.prompt_frame_index,
                bool(args.force),
                q,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    remaining = len(video_tasks)
    ok = 0
    err = 0
    skipped = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(video_tasks), desc="Generating masks", unit="video")

    while remaining > 0:
        status, vp, info, gpu_id = q.get()
        remaining -= 1
        if status == "ok":
            ok += 1
            msg = f"[OK][gpu{gpu_id}] {vp} -> {info}"
        elif status == "skip":
            skipped += 1
            msg = f"[SKIP][gpu{gpu_id}] {vp} -> {info}"
        else:
            err += 1
            msg = f"[ERR][gpu{gpu_id}] {vp} -> {info}"
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(ok=ok, skip=skipped, err=err)
            pbar.write(msg)
        else:
            print(msg)

    for p in procs:
        p.join()

    if pbar is not None:
        pbar.close()

    if err:
        raise SystemExit(f"Done with errors: ok={ok}, skip={skipped}, err={err}")
    print(f"Done: ok={ok}, skip={skipped}, err={err}")


if __name__ == "__main__":
    main()
