#!/usr/bin/env python3
"""Pi3X warp: estimate fixed source->target pose from first frame pair,
then reconstruct source video frame-by-frame and project into target view.

This script intentionally uses Pi3 utilities for geometry core steps:
- se3_inverse
- geotrf
- depth_edge
- recover_intrinsic_from_rays_d
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge, geotrf, recover_intrinsic_from_rays_d, se3_inverse


# ================================================================
# User-editable constants
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_VIDEO = str(PROJECT_ROOT / "assets" / "example" / "1.mp4")
TGT_VIDEO = str(PROJECT_ROOT / "assets" / "example" / "2.mp4")
OUT_DIR = str(PROJECT_ROOT / "outputs" / "pi3_warp_firstframe_pose")

DEVICE = "cuda"
MAX_FRAMES = 0  # 0 = full source/target overlap
CONF_THRES = 0.10
ENABLE_SCALE_ALIGN = True
PI3X_DEFAULT_MODEL = "yyfz233/Pi3X"
PI3X_DEFAULT_CACHE_ROOT = Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface"))


@dataclass
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    first_frame_bgr: np.ndarray


def read_video_meta_and_first_frame(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first = cap.read()
    cap.release()

    if not ok or first is None:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    if width <= 0 or height <= 0:
        height, width = first.shape[:2]

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    return VideoMeta(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
        first_frame_bgr=first,
    )


def load_source_video_tensor(video_path: Path) -> torch.Tensor:
    imgs = load_images_as_tensor(str(video_path), interval=1, verbose=True)
    if imgs.numel() == 0 or imgs.shape[0] == 0:
        raise RuntimeError(f"No frames loaded from source video: {video_path}")

    if MAX_FRAMES > 0:
        imgs = imgs[:MAX_FRAMES]
    return imgs.contiguous()


def preprocess_target_first_frame_like_source(
    frame_bgr: np.ndarray,
    source_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = source_hw
    resized = cv2.resize(frame_bgr, (src_w, src_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    return tensor.permute(2, 0, 1).contiguous()


def model_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # cuRoPE2D CUDA kernel in this environment does not support bfloat16.
    return torch.float16


def resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"

    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(device)
    return device


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def get_pi3x_model_source() -> str:
    return (
        os.environ.get("PI3X_CKPT")
        or os.environ.get("PI3X_MODEL")
        or os.environ.get("PI3X_MODEL_ID")
        or PI3X_DEFAULT_MODEL
    )


def _normalize_hf_cache_dir(path: Path) -> Path:
    path = path.expanduser()
    if (path / "hub").is_dir():
        return path / "hub"
    return path


def get_pi3x_cache_dir() -> Path | None:
    override = os.environ.get("PI3X_CACHE_DIR") or os.environ.get("PI3X_HUB_CACHE")
    if override:
        return _normalize_hf_cache_dir(Path(override))
    if PI3X_DEFAULT_CACHE_ROOT.is_dir():
        return _normalize_hf_cache_dir(PI3X_DEFAULT_CACHE_ROOT)
    return None


def _is_path_like_model_source(source: str) -> bool:
    source = source.strip()
    suffix = Path(source).suffix.lower()
    return (
        source.startswith(("/", "./", "../", "~"))
        or suffix in {".safetensors", ".pt", ".pth", ".bin"}
    )


def _resolve_local_model_path(source: str) -> Path | None:
    path = Path(source).expanduser()
    if path.exists():
        return path
    if _is_path_like_model_source(source):
        raise FileNotFoundError(f"Pi3X model path does not exist: {path}")
    return None


def _offline_or_local_only_requested() -> bool:
    return (
        _env_flag("PI3X_LOCAL_FILES_ONLY")
        or _env_flag("HF_HUB_OFFLINE")
        or _env_flag("TRANSFORMERS_OFFLINE")
    )


def _missing_model_message(source: str) -> str:
    cache_dir = get_pi3x_cache_dir()
    cache_hint = f" Current cache_dir={cache_dir}." if cache_dir is not None else ""
    return (
        f"Pi3X weights are not available locally for '{source}'. "
        "Either run once with network access so Hugging Face can cache yyfz233/Pi3X, "
        "or download model.safetensors manually and set PI3X_CKPT=/path/to/model.safetensors."
        f"{cache_hint}"
    )


def _download_lock_path(source: str) -> Path:
    cache_dir = get_pi3x_cache_dir()
    if cache_dir is not None:
        cache_root = cache_dir
    else:
        cache_root = _normalize_hf_cache_dir(
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        )
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in source)
    return cache_root / ".locks" / f"pi3x-{safe_name}.lock"


@contextmanager
def _pi3x_download_lock(source: str):
    lock_path = _download_lock_path(source)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_pi3x_from_checkpoint(ckpt_path: Path, device: torch.device) -> Pi3X:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Pi3X checkpoint file not found: {ckpt_path}")

    print(f"[model] Load Pi3X checkpoint: {ckpt_path}", flush=True)
    model = Pi3X().eval()
    if ckpt_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path), device="cpu")
    else:
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        "[model] Loaded Pi3X checkpoint "
        f"(missing_keys={len(incompatible.missing_keys)}, "
        f"unexpected_keys={len(incompatible.unexpected_keys)}).",
        flush=True,
    )
    return model.to(device).eval()


def ensure_pi3x_weights_available() -> None:
    """Warm the Hugging Face cache before spawning GPU workers."""
    source = get_pi3x_model_source()
    local_path = _resolve_local_model_path(source)
    if local_path is not None:
        if local_path.is_dir() and not (local_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Pi3X local directory lacks model.safetensors: {local_path}")
        if local_path.is_file():
            print(f"[model] Pi3X checkpoint configured: {local_path}", flush=True)
        else:
            print(f"[model] Pi3X local model directory configured: {local_path}", flush=True)
        return

    from huggingface_hub import hf_hub_download

    cache_dir = get_pi3x_cache_dir()
    try:
        hf_hub_download(
            repo_id=source,
            filename="model.safetensors",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print(f"[model] Pi3X weights already cached: {source} cache_dir={cache_dir}", flush=True)
        return
    except Exception as cache_exc:
        if _offline_or_local_only_requested():
            raise RuntimeError(_missing_model_message(source)) from cache_exc

    with _pi3x_download_lock(source):
        try:
            hf_hub_download(
                repo_id=source,
                filename="model.safetensors",
                cache_dir=cache_dir,
                local_files_only=True,
            )
            print(f"[model] Pi3X weights already cached: {source} cache_dir={cache_dir}", flush=True)
            return
        except Exception:
            pass

        print(f"[model] Pi3X weights are not cached; downloading once: {source} cache_dir={cache_dir}", flush=True)
        try:
            hf_hub_download(repo_id=source, filename="model.safetensors", cache_dir=cache_dir)
        except Exception as download_exc:
            raise RuntimeError(_missing_model_message(source)) from download_exc
        print(f"[model] Pi3X weights downloaded: {source}", flush=True)


def load_pi3x_model(device: torch.device) -> Pi3X:
    source = get_pi3x_model_source()
    cache_dir = get_pi3x_cache_dir()
    local_path = _resolve_local_model_path(source)
    if local_path is not None:
        if local_path.is_dir():
            print(f"[model] Load Pi3X from local directory: {local_path}", flush=True)
            return Pi3X.from_pretrained(str(local_path), local_files_only=True).to(device).eval()
        return _load_pi3x_from_checkpoint(local_path, device)

    try:
        print(f"[model] Try local cache first: {source} cache_dir={cache_dir}", flush=True)
        model = Pi3X.from_pretrained(source, cache_dir=cache_dir, local_files_only=True).to(device).eval()
        print("[model] Loaded Pi3X from local cache.", flush=True)
        return model
    except Exception as exc:
        print(f"[model] Local cache load failed: {exc}", flush=True)
        if _offline_or_local_only_requested():
            raise RuntimeError(_missing_model_message(source)) from exc

    with _pi3x_download_lock(source):
        try:
            print(f"[model] Retry local cache after download lock: {source} cache_dir={cache_dir}", flush=True)
            model = Pi3X.from_pretrained(source, cache_dir=cache_dir, local_files_only=True).to(device).eval()
            print("[model] Loaded Pi3X from local cache.", flush=True)
            return model
        except Exception:
            pass

        print(f"[model] Fallback to online download: {source} cache_dir={cache_dir}", flush=True)
        try:
            model = Pi3X.from_pretrained(source, cache_dir=cache_dir).to(device).eval()
        except Exception as online_exc:
            raise RuntimeError(_missing_model_message(source)) from online_exc
        print("[model] Loaded Pi3X from online source.", flush=True)
        return model


def run_pi3x_infer_source_sequence(
    model: Pi3X,
    src_imgs_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    src_input = src_imgs_cpu.unsqueeze(0).to(device)  # [1, N, 3, H, W]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=dtype):
            pred = model(imgs=src_input)

    conf_prob = torch.sigmoid(pred["conf"])[..., 0]
    local_depth = pred["local_points"][..., 2]
    edge_mask = depth_edge(local_depth, rtol=0.03)

    out = {
        "points": pred["points"][0].float().cpu(),
        "camera_poses": pred["camera_poses"][0].float().cpu(),
        "local_depth": local_depth[0].float().cpu(),
        "local_points": pred["local_points"][0].float().cpu(),
        "conf_prob": conf_prob[0].float().cpu(),
        "edge_mask": edge_mask[0].cpu(),
    }
    del pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run_pi3x_infer_first_pair(
    model: Pi3X,
    src_first_chw: torch.Tensor,
    tgt_first_chw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    pair = torch.stack([src_first_chw, tgt_first_chw], dim=0).unsqueeze(0).to(device)  # [1,2,3,H,W]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=dtype):
            pred = model(imgs=pair)

    out = {
        "camera_poses": pred["camera_poses"][0].float().cpu(),  # [2,4,4] c2w
        "local_points": pred["local_points"][0].float().cpu(),  # [2,H,W,3]
        "local_depth": pred["local_points"][0, ..., 2].float().cpu(),  # [2,H,W]
    }
    del pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def compute_fixed_tgt_from_src_transform(pair_c2w: torch.Tensor) -> torch.Tensor:
    # pair_c2w: [2,4,4]
    pair_w2c = se3_inverse(pair_c2w.float())  # [2,4,4]
    w2c_src0 = pair_w2c[0]
    w2c_tgt0 = pair_w2c[1]
    t_tgt_from_src = w2c_tgt0 @ torch.linalg.inv(w2c_src0)
    return t_tgt_from_src.float()


def robust_depth_median(depth_map: torch.Tensor) -> float:
    d = depth_map.float().reshape(-1)
    valid = torch.isfinite(d) & (d > 1e-6)
    if valid.any():
        return float(torch.median(d[valid]).item())
    return 1.0


def recover_target_intrinsics_from_pair(local_points_tgt: torch.Tensor) -> torch.Tensor:
    # local_points_tgt: [H, W, 3]
    rays_d = F.normalize(local_points_tgt.float(), dim=-1)
    k = recover_intrinsic_from_rays_d(
        rays_d[None], force_center_principal_point=True
    )[0]
    return k.float()


def scale_intrinsics(
    k: np.ndarray,
    from_hw: Tuple[int, int],
    to_hw: Tuple[int, int],
) -> np.ndarray:
    h0, w0 = from_hw
    h1, w1 = to_hw
    sx = float(w1) / float(w0)
    sy = float(h1) / float(h0)

    out = k.astype(np.float32).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def project_with_pi3_geotrf_and_zbuffer(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    valid_mask: np.ndarray,
    w2c_tgt: np.ndarray,
    k_tgt: np.ndarray,
    out_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    out_h, out_w = out_hw

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    depth = np.full((out_h, out_w), np.inf, dtype=np.float32)

    if points_world.size == 0:
        return rgb, depth

    finite_pts = np.isfinite(points_world).all(axis=-1)
    finite_col = np.isfinite(colors_rgb).all(axis=-1)
    valid = finite_pts & finite_col & valid_mask
    if not np.any(valid):
        return rgb, depth

    pts = points_world[valid]
    cols = colors_rgb[valid]

    cam_pts = geotrf(w2c_tgt[:3, :], pts, ncol=3, norm=False)
    z = cam_pts[:, 2]

    proj = k_tgt @ w2c_tgt[:3, :]
    uv = geotrf(proj, pts, ncol=2, norm=1)

    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)

    in_front = z > 1e-6
    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    ok = in_front & in_bounds & np.isfinite(z)

    if not np.any(ok):
        return rgb, depth

    u = u[ok]
    v = v[ok]
    z = z[ok]
    cols = cols[ok]
    order = np.argsort(z, kind="mergesort")
    u = u[order]
    v = v[order]
    z = z[order]
    cols = cols[order]

    flat = v * out_w + u
    _, first_idx = np.unique(flat, return_index=True)

    chosen_flat = flat[first_idx]
    chosen_z = z[first_idx]
    chosen_cols = np.clip(cols[first_idx] * 255.0, 0, 255).astype(np.uint8)

    depth.reshape(-1)[chosen_flat] = chosen_z
    rgb.reshape(-1, 3)[chosen_flat] = chosen_cols
    return rgb, depth


def render_sequence_and_write_videos(
    src_imgs_cpu: torch.Tensor,
    src_pred: Dict[str, torch.Tensor],
    t_tgt_from_src: torch.Tensor,
    k_tgt_render: np.ndarray,
    render_hw: Tuple[int, int],
    output_hw: Tuple[int, int],
    num_frames: int,
    out_dir: Path,
    out_fps: float,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_out = out_dir / "warped_rgb.mp4"
    mask_out = out_dir / "warped_mask.mp4"

    rgb_writer = imageio.get_writer(str(rgb_out), fps=float(out_fps), macro_block_size=None)
    mask_writer = imageio.get_writer(str(mask_out), fps=float(out_fps), macro_block_size=None)

    w2c_src_seq = se3_inverse(src_pred["camera_poses"].float())  # [N,4,4]
    t_tgt_from_src_np = t_tgt_from_src.detach().cpu().numpy().astype(np.float32)

    coverages = []

    points_seq = src_pred["points"][:num_frames].reshape(num_frames, -1, 3).numpy().astype(np.float32)
    colors_seq = (
        src_imgs_cpu[:num_frames]
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(num_frames, -1, 3)
        .numpy()
        .astype(np.float32)
    )

    try:
        for idx in range(num_frames):
            points_i = points_seq[idx]
            colors_i = colors_seq[idx]
            valid_i = np.ones(points_i.shape[0], dtype=bool)

            w2c_src_i = w2c_src_seq[idx].numpy().astype(np.float32)
            w2c_tgt_i = t_tgt_from_src_np @ w2c_src_i

            rgb_frame, depth_frame = project_with_pi3_geotrf_and_zbuffer(
                points_world=points_i,
                colors_rgb=colors_i,
                valid_mask=valid_i,
                w2c_tgt=w2c_tgt_i,
                k_tgt=k_tgt_render,
                out_hw=render_hw,
            )

            mask = (np.isfinite(depth_frame).astype(np.uint8) * 255)
            out_h, out_w = output_hw
            if render_hw != output_hw:
                rgb_frame = cv2.resize(rgb_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            mask_rgb = np.repeat(mask[..., None], 3, axis=2)

            rgb_writer.append_data(rgb_frame)
            mask_writer.append_data(mask_rgb)

            coverages.append(float(np.mean(mask > 0)))

            if (idx + 1) % 10 == 0 or (idx + 1) == num_frames:
                print(f"[render] {idx + 1}/{num_frames}", flush=True)
    finally:
        rgb_writer.close()
        mask_writer.close()

    stats = {
        "mask_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "mask_coverage_min": float(np.min(coverages)) if coverages else 0.0,
        "mask_coverage_max": float(np.max(coverages)) if coverages else 0.0,
    }

    return stats


def main() -> int:
    src_video = Path(SRC_VIDEO).expanduser().resolve()
    tgt_video = Path(TGT_VIDEO).expanduser().resolve()
    out_dir = Path(OUT_DIR).expanduser().resolve()

    if not src_video.is_file():
        raise FileNotFoundError(f"Source video not found: {src_video}")
    if not tgt_video.is_file():
        raise FileNotFoundError(f"Target video not found: {tgt_video}")

    print("[1/7] Read video metadata...")
    src_meta = read_video_meta_and_first_frame(src_video)
    tgt_meta = read_video_meta_and_first_frame(tgt_video)

    print("[2/7] Load source video tensor (Pi3 loader)...")
    src_imgs_cpu = load_source_video_tensor(src_video)

    src_n = int(src_imgs_cpu.shape[0])
    tgt_n = int(tgt_meta.frame_count) if tgt_meta.frame_count > 0 else src_n
    num_frames = min(src_n, tgt_n)
    if MAX_FRAMES > 0:
        num_frames = min(num_frames, MAX_FRAMES)

    if num_frames <= 0:
        raise RuntimeError("No overlapping frames available for rendering.")

    src_imgs_cpu = src_imgs_cpu[:num_frames].contiguous()
    model_h, model_w = int(src_imgs_cpu.shape[-2]), int(src_imgs_cpu.shape[-1])

    print(
        "[meta] "
        f"src={src_meta.width}x{src_meta.height}, fps={src_meta.fps:.3f}, frames={src_meta.frame_count}; "
        f"tgt={tgt_meta.width}x{tgt_meta.height}, fps={tgt_meta.fps:.3f}, frames={tgt_meta.frame_count}; "
        f"used_frames={num_frames}; model_size={model_w}x{model_h}"
    )

    print("[3/7] Prepare target first frame in source model resolution...")
    tgt_first_chw = preprocess_target_first_frame_like_source(
        tgt_meta.first_frame_bgr,
        source_hw=(model_h, model_w),
    )

    device = resolve_device(DEVICE)
    dtype = model_dtype_for_device(device)

    print(f"[4/7] Load Pi3X on {device}...")
    model = load_pi3x_model(device)

    print("[5/7] Infer source sequence...")
    src_pred = run_pi3x_infer_source_sequence(
        model=model,
        src_imgs_cpu=src_imgs_cpu,
        device=device,
        dtype=dtype,
    )

    print("[6/7] Infer first frame pair and estimate fixed source->target pose...")
    pair_pred = run_pi3x_infer_first_pair(
        model=model,
        src_first_chw=src_imgs_cpu[0],
        tgt_first_chw=tgt_first_chw,
        device=device,
        dtype=dtype,
    )

    t_tgt_from_src = compute_fixed_tgt_from_src_transform(pair_pred["camera_poses"])  # [4,4]

    depth_src_seq_median = robust_depth_median(src_pred["local_depth"][0])
    depth_src_pair_median = robust_depth_median(pair_pred["local_depth"][0])
    scale_raw = depth_src_seq_median / max(depth_src_pair_median, 1e-6)
    scale_used = float(np.clip(scale_raw, 0.25, 4.0))

    if ENABLE_SCALE_ALIGN:
        t_tgt_from_src = t_tgt_from_src.clone()
        t_tgt_from_src[:3, 3] *= scale_used

    k_tgt_model = recover_target_intrinsics_from_pair(pair_pred["local_points"][1])
    k_tgt_render = k_tgt_model.numpy().astype(np.float32)
    k_tgt_out = scale_intrinsics(
        k_tgt_render,
        from_hw=(model_h, model_w),
        to_hw=(tgt_meta.height, tgt_meta.width),
    )

    print(
        "[pose] "
        f"scale_raw={scale_raw:.4f}, "
        f"scale_used={scale_used:.4f}, "
        f"scale_align={'on' if ENABLE_SCALE_ALIGN else 'off'}"
    )

    print("[7/7] Render sequence and write videos...")
    render_stats = render_sequence_and_write_videos(
        src_imgs_cpu=src_imgs_cpu,
        src_pred=src_pred,
        t_tgt_from_src=t_tgt_from_src,
        k_tgt_render=k_tgt_render,
        render_hw=(model_h, model_w),
        output_hw=(tgt_meta.height, tgt_meta.width),
        num_frames=num_frames,
        out_dir=out_dir,
        out_fps=src_meta.fps,
    )

    meta = {
        "source_video": str(src_video),
        "target_video": str(tgt_video),
        "out_dir": str(out_dir),
        "device": str(device),
        "model": "Pi3X",
        "max_frames_config": int(MAX_FRAMES),
        "frames_used": int(num_frames),
        "source_fps_used_for_output": float(src_meta.fps),
        "source_meta": {
            "width": int(src_meta.width),
            "height": int(src_meta.height),
            "fps": float(src_meta.fps),
            "frame_count": int(src_meta.frame_count),
        },
        "target_meta": {
            "width": int(tgt_meta.width),
            "height": int(tgt_meta.height),
            "fps": float(tgt_meta.fps),
            "frame_count": int(tgt_meta.frame_count),
        },
        "model_resolution_hw": [int(model_h), int(model_w)],
        "output_resolution_hw": [int(tgt_meta.height), int(tgt_meta.width)],
        "enable_scale_align": bool(ENABLE_SCALE_ALIGN),
        "scale_ref_seq_over_pair_raw": float(scale_raw),
        "scale_ref_seq_over_pair_used": float(scale_used),
        "T_tgt_from_src_4x4": t_tgt_from_src.numpy().tolist(),
        "K_tgt_model_3x3": k_tgt_model.numpy().tolist(),
        "K_tgt_output_3x3": k_tgt_out.tolist(),
        "outputs": {
            "warped_rgb": str(out_dir / "warped_rgb.mp4"),
            "warped_mask": str(out_dir / "warped_mask.mp4"),
            "meta": str(out_dir / "meta.json"),
        },
        **render_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote: {out_dir / 'warped_rgb.mp4'}")
    print(f"[done] wrote: {out_dir / 'warped_mask.mp4'}")
    print(f"[done] wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
