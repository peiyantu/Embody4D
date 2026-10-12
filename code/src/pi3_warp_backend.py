from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.utils import (
    Warper,
    generate_traj_specified,
    generate_traj_txt,
    read_video_frames,
    save_video,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PI3_REPO = PROJECT_ROOT / "third_party" / "Pi3"
DEFAULT_PI3_REPO = str(PROJECT_PI3_REPO)
PI3_DEPTH_WARPER_BACKEND = "pi3_depth_warper"


def _resolve_pi3_repo(path: str | None) -> Path:
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get("PI3_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(PROJECT_PI3_REPO)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if (candidate / "run_pi3_warp_two_videos_firstframe_pose.py").is_file():
            return candidate.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Pi3 backend requested, but Pi3 repo was not found. "
        "Set --pi3_repo_path or PI3_REPO_PATH, or run scripts/setup_pi3.sh. Tried: " + tried
    )


def _import_pi3_core(pi3_repo: Path):
    repo_str = str(pi3_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    import run_pi3_warp_two_videos_firstframe_pose as core

    return core


def _set_optional_env(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


def _normalize_device_arg(device_arg: str | None) -> str:
    if device_arg is None:
        return "cuda:0"
    normalized = str(device_arg).strip().lower()
    if normalized.startswith("gpu") and normalized[3:].isdigit():
        return f"cuda:{normalized[3:]}"
    return normalized


def _frame_limit(opts) -> int:
    limits = []
    video_length = int(getattr(opts, "video_length", 0) or 0)
    pi3_max_frames = int(getattr(opts, "pi3_max_frames", 0) or 0)
    if video_length > 0:
        limits.append(video_length)
    if pi3_max_frames > 0:
        limits.append(pi3_max_frames)
    return min(limits) if limits else -1


def _load_warper_frames(src_video: Path, opts, limit: int) -> torch.Tensor:
    process_length = limit if limit > 0 else -1
    frames_np = read_video_frames(
        str(src_video),
        process_length,
        int(getattr(opts, "stride", 1) or 1),
        int(getattr(opts, "max_res", 1024) or 1024),
    )
    if limit > 0:
        frames_np = frames_np[:limit]
    if frames_np.shape[0] <= 0:
        raise RuntimeError(f"No frames loaded from source video: {src_video}")
    return torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() * 2.0 - 1.0


def _load_pi3_frames(core, src_video: Path, opts, limit: int) -> torch.Tensor:
    stride = max(1, int(getattr(opts, "stride", 1) or 1))
    imgs = core.load_images_as_tensor(str(src_video), interval=stride, verbose=True)
    if imgs.numel() == 0 or imgs.shape[0] == 0:
        raise RuntimeError(f"No frames loaded for Pi3 reconstruction: {src_video}")
    if limit > 0:
        imgs = imgs[:limit]
    return imgs.contiguous()


def _sanitize_depths(depths: torch.Tensor, near: float, far: float) -> torch.Tensor:
    depths = depths.float()
    valid = torch.isfinite(depths) & (depths > 1e-6)
    if valid.any():
        fill = torch.median(depths[valid])
    else:
        fill = torch.tensor(1.0, dtype=depths.dtype, device=depths.device)
    depths = torch.where(valid, depths, fill)
    min_depth = max(float(near), 1e-5)
    max_depth = max(float(far), min_depth)
    return depths.clamp(min=min_depth, max=max_depth)


def _pi3_depths_for_warper(
    src_pred: Dict[str, torch.Tensor],
    num_frames: int,
    target_hw: Tuple[int, int],
    opts,
) -> torch.Tensor:
    depths = src_pred["local_depth"][:num_frames].unsqueeze(1)
    depths = _sanitize_depths(
        depths,
        near=float(getattr(opts, "near", 1e-4) or 1e-4),
        far=float(getattr(opts, "far", 10000.0) or 10000.0),
    )
    if tuple(depths.shape[-2:]) != tuple(target_hw):
        depths = F.interpolate(
            depths,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return _sanitize_depths(
        depths,
        near=float(getattr(opts, "near", 1e-4) or 1e-4),
        far=float(getattr(opts, "far", 10000.0) or 10000.0),
    )


def _build_poses(
    opts,
    depths: torch.Tensor,
    num_frames: int,
    device: torch.device,
    f_new: float | None = None,
):
    h, w = int(depths.shape[-2]), int(depths.shape[-1])
    center_depth = float(depths[0, 0, h // 2, w // 2].detach().cpu())
    radius = min(center_depth * float(getattr(opts, "radius_scale", 1.0)), 5.0)

    cx = float(w) / 2.0
    cy = float(h) / 2.0
    fx = 500.0 * float(w) / 1024.0
    fy = 500.0 * float(h) / 576.0
    if f_new is None:
        K = (
            torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(device)
        )
    else:
        fx_new = float(f_new) * float(w) / 1024.0
        fy_new = float(f_new) * float(h) / 576.0
        fx_values = torch.linspace(fx, fx_new, num_frames, device=device)
        fy_values = torch.linspace(fy, fy_new, num_frames, device=device)
        K = torch.zeros((num_frames, 3, 3), device=device)
        K[:, 0, 0] = fx_values
        K[:, 1, 1] = fy_values
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0

    c2w_init = (
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        .to(device)
        .unsqueeze(0)
    )

    if opts.camera == "target":
        if getattr(opts, "target_pose", None) is None:
            raise ValueError("--camera target requires --target_pose.")
        dtheta, dphi, dr, dx, dy = opts.target_pose
        poses = generate_traj_specified(
            c2w_init,
            dtheta,
            dphi,
            dr * radius,
            dx,
            dy,
            num_frames,
            device,
            direct=getattr(opts, "direct_pose", False),
        )
    elif opts.camera == "traj":
        with open(opts.traj_txt, "r", encoding="utf-8") as file:
            lines = file.readlines()
        theta = [float(i) for i in lines[0].split()]
        phi = [float(i) for i in lines[1].split()]
        r = [float(i) * radius for i in lines[2].split()]
        poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, device)
    else:
        raise ValueError(f"Unsupported camera mode: {opts.camera}")

    poses[:, 2, 3] = poses[:, 2, 3] + radius
    if getattr(opts, "direct_pose", False):
        pose_s = poses[0:1].repeat(num_frames, 1, 1)
        pose_t = poses[1:2].repeat(num_frames, 1, 1)
    else:
        anchor_idx = int(getattr(opts, "anchor_idx", 0) or 0)
        pose_s = poses[anchor_idx : anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
    return pose_s, pose_t, K, radius


def _render_with_depth_warper(
    frames: torch.Tensor,
    depths: torch.Tensor,
    opts,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    warper = Warper(device=opts.device)
    device = warper.device
    frames = frames.to(device)
    depths = depths.to(device)

    mode = getattr(opts, "mode", "gradual")
    num_frames = int(frames.shape[0])
    use_first = bool(getattr(opts, "use_first_frame_mask", False))
    warped_images = []
    masks = []

    with torch.no_grad():
        if mode == "direct":
            pose_s, pose_t, K, radius = _build_poses(
                opts, depths, num_frames=2, device=device
            )
            pose_target = pose_t[-1:]
            for i in tqdm(range(num_frames), desc="Pi3 depth warp"):
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame, mask, _, _ = warper.forward_warp(
                    frames[i : i + 1],
                    None,
                    depth_i,
                    pose_s[0:1],
                    pose_target,
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame.detach().cpu())
                masks.append(mask.detach().cpu())
        elif mode == "bullet":
            pose_s, pose_t, K, radius = _build_poses(
                opts, depths, num_frames=num_frames, device=device
            )
            for i in tqdm(range(num_frames), desc="Pi3 depth warp"):
                depth_i = depths[0:1] if use_first else depths[-1:]
                warped_frame, mask, _, _ = warper.forward_warp(
                    frames[-1:],
                    None,
                    depth_i,
                    pose_s[0:1],
                    pose_t[i : i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame.detach().cpu())
                masks.append(mask.detach().cpu())
        elif mode == "zoom":
            pose_s, pose_t, K, radius = _build_poses(
                opts, depths, num_frames=num_frames, device=device, f_new=250.0
            )
            for i in tqdm(range(num_frames), desc="Pi3 depth warp"):
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame, mask, _, _ = warper.forward_warp(
                    frames[i : i + 1],
                    None,
                    depth_i,
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[0:1],
                    K[i : i + 1],
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame.detach().cpu())
                masks.append(mask.detach().cpu())
        elif mode == "gradual":
            pose_s, pose_t, K, radius = _build_poses(
                opts, depths, num_frames=num_frames, device=device
            )
            for i in tqdm(range(num_frames), desc="Pi3 depth warp"):
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame, mask, _, _ = warper.forward_warp(
                    frames[i : i + 1],
                    None,
                    depth_i,
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[i : i + 1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame.detach().cpu())
                masks.append(mask.detach().cpu())
        else:
            raise ValueError(f"Unsupported inference mode for Pi3 warp: {mode}")

    cond_video = (torch.cat(warped_images) + 1.0) / 2.0
    cond_masks = torch.cat(masks)
    if use_first:
        cond_masks = masks[0].repeat(num_frames, 1, 1, 1)

    render_stats = {
        "radius_used": float(radius),
        "mask_coverage_mean": float(cond_masks.mean().item()),
        "mask_coverage_min": float(cond_masks.flatten(1).mean(dim=1).min().item()),
        "mask_coverage_max": float(cond_masks.flatten(1).mean(dim=1).max().item()),
    }
    return cond_video.clamp(0.0, 1.0), cond_masks.clamp(0.0, 1.0), render_stats


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _maybe_reuse_existing(meta_out: Path, rgb_out: Path, mask_out: Path, opts) -> int | None:
    if not (
        getattr(opts, "pi3_reuse_warp", False)
        and rgb_out.is_file()
        and mask_out.is_file()
        and meta_out.is_file()
    ):
        return None
    with meta_out.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("backend") != PI3_DEPTH_WARPER_BACKEND:
        print(
            "Existing Pi3 warp files use a legacy backend; regenerating "
            f"{rgb_out.parent}.",
            flush=True,
        )
        return None
    return int(meta.get("frames_used", getattr(opts, "video_length", 0) or 0))


def generate_pi3_warp(opts) -> Tuple[str, str, int]:
    """Generate DepthAnything-style warp/mask videos using Pi3 depth."""
    src_video = Path(opts.video_path).expanduser().resolve()
    if not src_video.is_file():
        raise FileNotFoundError(f"Pi3 source video not found: {src_video}")

    pi3_repo = _resolve_pi3_repo(getattr(opts, "pi3_repo_path", None))
    core = _import_pi3_core(pi3_repo)

    _set_optional_env("PI3X_MODEL", getattr(opts, "pi3_model", None))
    _set_optional_env("PI3X_CKPT", getattr(opts, "pi3_ckpt", None))
    _set_optional_env("PI3X_CACHE_DIR", getattr(opts, "pi3_cache_dir", None))
    if getattr(opts, "pi3_local_files_only", False):
        os.environ["PI3X_LOCAL_FILES_ONLY"] = "1"

    out_dir_arg = getattr(opts, "pi3_output_dir", None)
    if out_dir_arg:
        out_dir = Path(out_dir_arg).expanduser().resolve()
    else:
        out_dir = Path(opts.save_dir).expanduser().resolve() / "pi3_warp"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_out = out_dir / "warped_rgb.mp4"
    mask_out = out_dir / "warped_mask.mp4"
    meta_out = out_dir / "meta.json"
    reused_frames = _maybe_reuse_existing(meta_out, rgb_out, mask_out, opts)
    if reused_frames is not None:
        return str(rgb_out), str(mask_out), reused_frames

    limit = _frame_limit(opts)
    frames = _load_warper_frames(src_video, opts, limit)
    src_imgs_cpu = _load_pi3_frames(core, src_video, opts, limit)
    num_frames = min(int(frames.shape[0]), int(src_imgs_cpu.shape[0]))
    if num_frames <= 0:
        raise RuntimeError("Pi3 backend found no source frames to render.")
    frames = frames[:num_frames].contiguous()
    src_imgs_cpu = src_imgs_cpu[:num_frames].contiguous()

    model_h, model_w = int(src_imgs_cpu.shape[-2]), int(src_imgs_cpu.shape[-1])
    frame_h, frame_w = int(frames.shape[-2]), int(frames.shape[-1])

    device = core.resolve_device(
        _normalize_device_arg(getattr(opts, "pi3_device", None) or opts.device)
    )
    dtype = core.model_dtype_for_device(device)
    model = core.load_pi3x_model(device)

    src_pred = core.run_pi3x_infer_source_sequence(
        model=model,
        src_imgs_cpu=src_imgs_cpu,
        device=device,
        dtype=dtype,
    )
    depths = _pi3_depths_for_warper(
        src_pred=src_pred,
        num_frames=num_frames,
        target_hw=(frame_h, frame_w),
        opts=opts,
    )

    cond_video, cond_masks, render_stats = _render_with_depth_warper(
        frames=frames,
        depths=depths,
        opts=opts,
    )

    out_fps = int(getattr(opts, "fps", 10) or 10)
    save_video(cond_video.permute(0, 2, 3, 1), str(rgb_out), fps=out_fps)
    save_video(
        cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
        str(mask_out),
        fps=out_fps,
    )

    depth_cpu = depths.detach().cpu()
    meta = {
        "backend": PI3_DEPTH_WARPER_BACKEND,
        "pi3_repo": str(pi3_repo),
        "source_video": str(src_video),
        "frames_used": int(num_frames),
        "source_fps_used_for_output": float(out_fps),
        "stride": int(getattr(opts, "stride", 1) or 1),
        "mode": getattr(opts, "mode", None),
        "camera": getattr(opts, "camera", None),
        "target_pose": _json_value(getattr(opts, "target_pose", None)),
        "traj_txt": getattr(opts, "traj_txt", None),
        "direct_pose": bool(getattr(opts, "direct_pose", False)),
        "use_first_frame_mask": bool(getattr(opts, "use_first_frame_mask", False)),
        "mask_cleanup": bool(getattr(opts, "mask", False)),
        "radius_scale": float(getattr(opts, "radius_scale", 1.0) or 1.0),
        "pi3_model_resolution_hw": [int(model_h), int(model_w)],
        "warper_resolution_hw": [int(frame_h), int(frame_w)],
        "depth_min": float(depth_cpu.min().item()),
        "depth_max": float(depth_cpu.max().item()),
        "depth_median": float(torch.median(depth_cpu).item()),
        "outputs": {
            "warped_rgb": str(rgb_out),
            "warped_mask": str(mask_out),
            "meta": str(meta_out),
        },
        **render_stats,
    }
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    del model, src_pred, frames, depths, cond_video, cond_masks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(rgb_out), str(mask_out), int(num_frames)
