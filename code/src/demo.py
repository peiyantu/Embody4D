import gc
import json
import os
import torch
from models.infer import DepthCrafterDemo
import numpy as np
import torch
from transformers import T5EncoderModel
from omegaconf import OmegaConf
from PIL import Image
from models.crosstransformer3d import CrossTransformer3DModel
from models.autoencoder_magvit import AutoencoderKLCogVideoX
from models.latent_residual_unet import WarpMaskLatentResidualUNet
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from models.utils import *
from diffusers import (
    AutoencoderKL,
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def resolve_latent_residual_checkpoint(path):
    candidates = [path]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(path):
        candidates.append(os.path.join(script_dir, path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        direct = os.path.join(candidate, "latent_residual_mse.pt")
        if os.path.isfile(direct):
            return direct
        if not os.path.isdir(candidate):
            continue

        checkpoints = []
        for child_name in os.listdir(candidate):
            child = os.path.join(candidate, child_name)
            ckpt = os.path.join(child, "latent_residual_mse.pt")
            if not os.path.isdir(child) or not os.path.isfile(ckpt):
                continue
            if child_name == "final":
                step = 10**12
            elif child_name.startswith("checkpoint-"):
                try:
                    step = int(child_name.split("-", 1)[1])
                except ValueError:
                    step = -1
            else:
                step = -1
            checkpoints.append((step, ckpt))
        if checkpoints:
            checkpoints.sort(key=lambda item: item[0])
            return checkpoints[-1][1]

    raise FileNotFoundError(f"no latent_residual_mse.pt found under: {path}")


def load_latent_residual_model(path, device, dtype):
    ckpt_path = resolve_latent_residual_checkpoint(path)
    payload = torch.load(ckpt_path, map_location="cpu")
    config = payload.get("model_config") or {}
    if not config:
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)

    model = WarpMaskLatentResidualUNet(**config)
    state_dict = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"latent residual checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, ckpt_path


class TrajCrafter:
    def __init__(self, opts, gradio=False):
        if self._use_external_warp_mask(opts):
            self.funwarp = None
            self.depth_estimater = None
        else:
            self.funwarp = Warper(device=opts.device)
            # self.depth_estimater = VDADemo(pre_train_path=opts.pre_train_path_vda,device=opts.device)
            self.depth_estimater = DepthCrafterDemo(
                unet_path=opts.unet_path,
                pre_train_path=opts.pre_train_path,
                cpu_offload=opts.cpu_offload,
                device=opts.device,
            )
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(
            opts.blip_path, torch_dtype=torch.float16
        ).to(opts.device)
        self.setup_diffusion(opts)
        if gradio:
            self.opts = opts

    def _use_external_warp_mask(self, opts):
        return bool(getattr(opts, 'warp_path', None) and getattr(opts, 'mask_path', None))

    def _latent_routing_kwargs(self, opts):
        if getattr(self, "latent_residual_model", None) is None:
            return {}
        return {
            "latent_residual_model": self.latent_residual_model,
            "latent_route_top_k": getattr(opts, "latent_route_top_k", 2),
            "latent_route_secondary_weight": getattr(opts, "latent_route_secondary_weight", 0.2),
            "latent_route_copy_valid_threshold": getattr(opts, "latent_route_copy_valid_threshold", 0.85),
            "latent_route_copy_delta_threshold": getattr(opts, "latent_route_copy_delta_threshold", 0.35),
            "latent_route_inpaint_delta_threshold": getattr(opts, "latent_route_inpaint_delta_threshold", 0.50),
            "latent_route_inpaint_valid_threshold": getattr(opts, "latent_route_inpaint_valid_threshold", 0.45),
            "latent_route_force_inpaint_delta_threshold": getattr(
                opts, "latent_route_force_inpaint_delta_threshold", 0.85
            ),
            "latent_route_postprocess_mode": getattr(opts, "latent_route_postprocess_mode", "legacy"),
            "latent_route_smooth_spatial_kernel": getattr(opts, "latent_route_smooth_spatial_kernel", 1),
            "latent_route_smooth_temporal_kernel": getattr(opts, "latent_route_smooth_temporal_kernel", 1),
            "latent_route_valid_smooth_spatial_kernel": getattr(
                opts, "latent_route_valid_smooth_spatial_kernel", 1
            ),
            "latent_route_vote_spatial_kernel": getattr(opts, "latent_route_vote_spatial_kernel", 1),
            "latent_route_vote_temporal_kernel": getattr(opts, "latent_route_vote_temporal_kernel", 1),
            "latent_route_postprocess_on_video_grid": getattr(
                opts, "latent_route_postprocess_on_video_grid", False
            ),
            "latent_ffn_expert_scale": getattr(opts, "latent_ffn_expert_scale", 1.0),
            "latent_route_copy_residual_scale": getattr(
                opts, "latent_route_copy_residual_scale", 1.0
            ),
        }

    def _load_external_warp_mask(self, opts):
        # read_video_frames 返回 [0, 1] float32；与内部 cond_video / cond_masks 范围一致
        cond_video_np = read_video_frames(
            opts.warp_path, opts.video_length, 1, opts.max_res
        )
        cond_mask_np = read_video_frames(
            opts.mask_path, opts.video_length, 1, opts.max_res
        )
        # warp: 保持 [0, 1]，与内部 (warped+1)/2 一致
        cond_video = (
            torch.from_numpy(cond_video_np[: opts.video_length].astype(np.float32))
            .permute(0, 3, 1, 2)
            .to(opts.device)
        )
        cond_video = self._fit_video_length(cond_video, opts.video_length)
        # mask: 二值化为 0/1，与 forward_warp 的 mask 一致，避免存盘时因量化产生灰阶
        cond_masks_rgb = (
            torch.from_numpy(cond_mask_np[: opts.video_length].astype(np.float32))
            .permute(0, 3, 1, 2)
            .to(opts.device)
        )
        cond_masks_rgb = self._fit_video_length(cond_masks_rgb, opts.video_length)
        cond_masks = (cond_masks_rgb[:, :1, :, :] > 0.5).float()
        return cond_video, cond_masks

    def _direct_resize_video_tensor(self, tensor, sample_size, mode="bilinear"):
        target_hw = (int(sample_size[0]), int(sample_size[1]))
        if mode == "nearest":
            return F.interpolate(tensor, size=target_hw, mode=mode)
        return F.interpolate(tensor, size=target_hw, mode=mode, align_corners=False)

    def _direct_resize_inference_triplet(self, frames, cond_video, cond_masks, sample_size):
        # Inference preprocessing uses direct resize only: no crop and no padding.
        frames = self._direct_resize_video_tensor(frames, sample_size, mode="bilinear")
        cond_video = self._direct_resize_video_tensor(cond_video, sample_size, mode="bilinear")
        cond_masks = self._direct_resize_video_tensor(cond_masks, sample_size, mode="nearest")
        return frames, cond_video, cond_masks

    def _fit_video_length(self, tensor, video_length):
        if tensor.shape[0] == video_length:
            return tensor
        if tensor.shape[0] > video_length:
            return tensor[:video_length]
        if tensor.shape[0] == 0:
            raise ValueError("video has no readable frames.")
        pad = tensor[-1:].repeat(video_length - tensor.shape[0], 1, 1, 1)
        return torch.cat([tensor, pad], dim=0)

    def _save_viz(self, left, right, save_path, fps, device):
        left = left.to(device)
        right = right.to(device)
        frame_count = min(left.shape[1], right.shape[1])
        if frame_count <= 0:
            return
        left = left[:, :frame_count]
        right = right[:, :frame_count]
        interval = torch.ones(
            left.shape[0],
            frame_count,
            left.shape[2],
            30,
            device=left.device,
            dtype=left.dtype,
        )
        result = torch.cat((left, interval, right), dim=3)
        result_reverse = torch.flip(result, dims=[1])
        final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
        save_video(final_result.permute(1, 2, 3, 0), save_path, fps=fps)

    def infer_gradual(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        # 使用实际读取的帧数（video_length=-1 表示自动，或实际帧数不足时调整）
        actual_frames = frames.shape[0]
        if opts.video_length <= 0 or actual_frames < opts.video_length:
            opts.video_length = actual_frames

        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        if self._use_external_warp_mask(opts):
            cond_video, cond_masks = self._load_external_warp_mask(opts)
        else:
            # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
            depths = self.depth_estimater.infer(
                frames,
                opts.near,
                opts.far,
                opts.depth_inference_steps,
                opts.depth_guidance_scale,
                window_size=opts.window_size,
                overlap=opts.overlap,
            ).to(opts.device)
            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
            warped_images = []
            masks = []
            use_first = getattr(opts, 'use_first_frame_mask', False)
            for i in tqdm(range(opts.video_length)):
                # use_first_frame_mask: 用第一帧深度计算 warp，mask 固定为第一帧；render 仍用每帧图像保持动态
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
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
                warped_images.append(warped_frame2)
                masks.append(mask2)
            cond_video = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks = torch.cat(masks)
            if use_first:
                cond_masks = masks[0].repeat(opts.video_length, 1, 1, 1)

        frames, cond_video, cond_masks = self._direct_resize_inference_triplet(
            frames, cond_video, cond_masks, opts.sample_size
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = self._build_frames_ref(opts, frames[:, :, :10, :, :])
        front_video = self._build_front_video(opts)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                front_video=front_video,
                **self._latent_routing_kwargs(opts),
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            self._save_viz(
                frames[0],
                sample[0],
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
                device=opts.device,
            )

    def infer_direct(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        actual_frames = frames.shape[0]
        if opts.video_length <= 0 or actual_frames < opts.video_length:
            opts.video_length = actual_frames

        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        if self._use_external_warp_mask(opts):
            cond_video, cond_masks = self._load_external_warp_mask(opts)
        else:
            # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
            depths = self.depth_estimater.infer(
                frames,
                opts.near,
                opts.far,
                opts.depth_inference_steps,
                opts.depth_guidance_scale,
                window_size=opts.window_size,
                overlap=opts.overlap,
            ).to(opts.device)
            # 用2帧插值只是为了拿到初始位姿和目标位姿
            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=2)
            pose_target = pose_t[-1:]  # 固定目标位姿

            warped_images = []
            masks = []
            use_first = getattr(opts, 'use_first_frame_mask', False)
            for i in tqdm(range(opts.video_length)):
                # use_first_frame_mask: 用第一帧深度计算 warp，mask 固定为第一帧；render 仍用每帧图像保持动态
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
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
                warped_images.append(warped_frame2)
                masks.append(mask2)
            cond_video = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks = torch.cat(masks)
            if use_first:
                cond_masks = masks[0].repeat(opts.video_length, 1, 1, 1)
        frames, cond_video, cond_masks = self._direct_resize_inference_triplet(
            frames, cond_video, cond_masks, opts.sample_size
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = self._build_frames_ref(opts, frames[:, :, :10, :, :])
        front_video = self._build_front_video(opts)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                front_video=front_video,
                **self._latent_routing_kwargs(opts),
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            self._save_viz(
                frames[0],
                sample[0],
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
                device=opts.device,
            )

    def infer_bullet(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        # 使用实际读取的帧数（video_length=-1 表示自动，或实际帧数不足时调整）
        actual_frames = frames.shape[0]
        if opts.video_length <= 0 or actual_frames < opts.video_length:
            opts.video_length = actual_frames

        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        if self._use_external_warp_mask(opts):
            cond_video, cond_masks = self._load_external_warp_mask(opts)
        else:
            # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
            depths = self.depth_estimater.infer(
                frames,
                opts.near,
                opts.far,
                opts.depth_inference_steps,
                opts.depth_guidance_scale,
                window_size=opts.window_size,
                overlap=opts.overlap,
            ).to(opts.device)
            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

            warped_images = []
            masks = []
            use_first = getattr(opts, 'use_first_frame_mask', False)
            for i in tqdm(range(opts.video_length)):
                # use_first_frame_mask: 用第一帧深度计算 warp，mask 固定为第一帧；render 仍用最后一帧 warp 保持动态
                depth_i = depths[0:1] if use_first else depths[-1:]
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
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
                warped_images.append(warped_frame2)
                masks.append(mask2)
            cond_video = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks = torch.cat(masks)
            if use_first:
                cond_masks = masks[0].repeat(opts.video_length, 1, 1, 1)
        frames, cond_video, cond_masks = self._direct_resize_inference_triplet(
            frames, cond_video, cond_masks, opts.sample_size
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = self._build_frames_ref(opts, frames[:, :, -10:, :, :])
        front_video = self._build_front_video(opts)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                front_video=front_video,
                **self._latent_routing_kwargs(opts),
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_left_full = torch.cat(
                [
                    tensor_left,
                    tensor_left[:, -1:, :, :].repeat(1, tensor_left.shape[1] - 1, 1, 1),
                ],
                dim=1,
            )
            tensor_right = sample[0].to(opts.device)
            tensor_right_full = torch.cat(
                [tensor_left, tensor_right[:, 1:, :, :]], dim=1
            )
            self._save_viz(
                tensor_left_full,
                tensor_right_full,
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 4,
                device=opts.device,
            )

    def infer_zoom(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        # 使用实际读取的帧数（video_length=-1 表示自动，或实际帧数不足时调整）
        actual_frames = frames.shape[0]
        if opts.video_length <= 0 or actual_frames < opts.video_length:
            opts.video_length = actual_frames

        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        if self._use_external_warp_mask(opts):
            cond_video, cond_masks = self._load_external_warp_mask(opts)
        else:
            # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
            depths = self.depth_estimater.infer(
                frames,
                opts.near,
                opts.far,
                opts.depth_inference_steps,
                opts.depth_guidance_scale,
                window_size=opts.window_size,
                overlap=opts.overlap,
            ).to(opts.device)
            pose_s, pose_t, K = self.get_poses_f(
                opts, depths, num_frames=opts.video_length, f_new=250
            )

            warped_images = []
            masks = []
            use_first = getattr(opts, 'use_first_frame_mask', False)
            for i in tqdm(range(opts.video_length)):
                # use_first_frame_mask: 用第一帧深度计算 warp，mask 固定为第一帧；render 仍用每帧图像保持动态
                depth_i = depths[0:1] if use_first else depths[i : i + 1]
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i : i + 1],
                    None,
                    depth_i,
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[0 : 1],
                    K[i : i + 1],
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
            cond_video = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks = torch.cat(masks)
            if use_first:
                cond_masks = masks[0].repeat(opts.video_length, 1, 1, 1)

        frames, cond_video, cond_masks = self._direct_resize_inference_triplet(
            frames, cond_video, cond_masks, opts.sample_size
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = self._build_frames_ref(opts, frames[:, :, :10, :, :])
        front_video = self._build_front_video(opts)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                front_video=front_video,
                **self._latent_routing_kwargs(opts),
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            self._save_viz(
                frames[0],
                sample[0],
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
                device=opts.device,
            )

    def _build_frames_ref(self, opts, default_frames_ref):
        """Return frames_ref tensor [1,3,10,H,W] in [0,1].
        Uses refvideo when opts.refvideo is set, otherwise falls back to default_frames_ref."""
        refvideo_path = getattr(opts, 'refvideo', None)
        if not refvideo_path:
            return default_frames_ref
        ref_np = read_video_frames(refvideo_path, 10, 1, opts.max_res)
        if ref_np.shape[0] < 10:
            raise ValueError(
                f"refvideo has only {ref_np.shape[0]} frames, need at least 10."
            )
        ref_tensor = (
            torch.from_numpy(ref_np[:10].astype(np.float32))
            .permute(0, 3, 1, 2)
            .to(opts.device)
        )  # [10, 3, H, W] in [0,1]
        ref_tensor = self._direct_resize_video_tensor(
            ref_tensor, opts.sample_size, mode="bilinear"
        )
        return ref_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, 10, H, W]

    def _build_front_video(self, opts):
        """Return optional front mask tensor [1,3,F,H,W] in [0,1]."""
        frontvideo_path = getattr(opts, 'frontvideo', None)
        if not frontvideo_path:
            if getattr(opts, 'enable_front_guidance', False) or getattr(
                opts, 'enable_front_condition', False
            ):
                raise ValueError(
                    "Front guidance/condition requires a valid `frontvideo` path."
                )
            return None
        front_length = (
            opts.video_length
            if getattr(opts, 'enable_front_condition', False)
            else min(10, opts.video_length)
        )
        front_np = read_video_frames(frontvideo_path, front_length, 1, opts.max_res)
        front_tensor = (
            torch.from_numpy(front_np[:front_length].astype(np.float32))
            .permute(0, 3, 1, 2)
            .to(opts.device)
        )
        front_tensor = self._fit_video_length(front_tensor, front_length)
        front_tensor = self._direct_resize_video_tensor(
            front_tensor, opts.sample_size, mode="nearest"
        )
        return front_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    def get_caption(self, opts, image):
        if getattr(opts, "prompt", None):
            return opts.prompt + opts.refine_prompt
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        h, w = int(depths.shape[-2]), int(depths.shape[-1])
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        fx = 500.0 * float(w) / 1024.0
        fy = 500.0 * float(h) / 576.0
        K = (
            torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device,
                direct=getattr(opts, 'direct_pose', False),
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        if getattr(opts, 'direct_pose', False):
            # 第0帧也完全目标视角：pose_s=初始位姿，pose_t=全部目标位姿
            pose_s = poses[0:1].repeat(num_frames, 1, 1)
            pose_t = poses[1:2].repeat(num_frames, 1, 1)
        else:
            pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
            pose_t = poses
        return pose_s, pose_t, K

    def get_poses_f(self, opts, depths, num_frames, f_new):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        h, w = int(depths.shape[-2]), int(depths.shape[-1])
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        fx = 500.0 * float(w) / 1024.0
        fy = 500.0 * float(h) / 576.0
        fx_new = float(f_new) * float(w) / 1024.0
        fy_new = float(f_new) * float(h) / 576.0
        # f_new,d_r: 250,0.5; 1000,-0.9
        fx_values = torch.linspace(fx, fx_new, num_frames, device=opts.device)
        fy_values = torch.linspace(fy, fy_new, num_frames, device=opts.device)
        K = torch.zeros((num_frames, 3, 3), device=opts.device)
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
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device,
                direct=getattr(opts, 'direct_pose', False),
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        if getattr(opts, 'direct_pose', False):
            pose_s = poses[0:1].repeat(num_frames, 1, 1)
            pose_t = poses[1:2].repeat(num_frames, 1, 1)
        else:
            pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
            pose_t = poses
        return pose_s, pose_t, K

    def setup_diffusion(self, opts):
        latent_residual_path = getattr(opts, "latent_residual_checkpoint", None)
        if latent_residual_path:
            transformer = CrossTransformer3DModel.from_pretrained_expert_aware(
                opts.transformer_path,
                ffn_expert_scale=getattr(opts, "latent_ffn_expert_scale", 1.0),
            ).to(opts.weight_dtype)
        else:
            # The released checkpoint contains optional FFN expert-router tensors.
            # When latent experts are disabled, load the matching base tensors and
            # ignore those optional router tensors so the default path stays usable.
            transformer = CrossTransformer3DModel.from_pretrained_cus(
                opts.transformer_path,
            ).to(opts.weight_dtype)
        self.latent_residual_model = None
        if latent_residual_path:
            self.latent_residual_model, latent_residual_ckpt = load_latent_residual_model(
                latent_residual_path,
                opts.device,
                opts.weight_dtype,
            )
            print(f"Loaded frozen latent residual router from {latent_residual_ckpt}")
        if hasattr(transformer, "set_front_guidance"):
            front_guidance_kwargs = {
                "enable": bool(getattr(opts, "enable_front_guidance", False)),
                "target_scale": float(getattr(opts, "front_guidance_target_scale", 0.6)),
                "boost_strength": float(getattr(opts, "front_guidance_boost_strength", 5.0)),
                "mode": str(getattr(opts, "front_guidance_mode", "blend")),
                "min_area": float(getattr(opts, "front_guidance_min_area", 0.0)),
                "max_area": float(getattr(opts, "front_guidance_max_area", 1.0)),
            }
            memory_high_threshold = getattr(opts, "front_memory_high_threshold", None)
            memory_quality_threshold = getattr(opts, "front_memory_quality_threshold", None)
            memory_noise_threshold = getattr(opts, "front_memory_noise_threshold", None)
            memory_min_alignment = getattr(opts, "front_memory_min_alignment", None)
            gate_floor = getattr(opts, "front_guidance_gate_floor", None)
            memory_quality_power = getattr(opts, "front_memory_quality_power", None)
            memory_residual_ratio_max = getattr(
                opts,
                "front_memory_residual_ratio_max",
                None,
            )
            memory_query_mask_power = getattr(
                opts,
                "front_memory_query_mask_power",
                None,
            )
            memory_query_mask_floor = getattr(
                opts,
                "front_memory_query_mask_floor",
                None,
            )
            memory_train_gate_gradient_multiplier = getattr(
                opts,
                "front_memory_train_gate_gradient_multiplier",
                None,
            )
            memory_max_area_std = getattr(opts, "front_memory_max_area_std", None)
            memory_min_area_std = getattr(opts, "front_memory_min_area_std", None)
            memory_min_valid_frame_ratio = getattr(
                opts,
                "front_memory_min_valid_frame_ratio",
                None,
            )
            memory_max_area_jump = getattr(opts, "front_memory_max_area_jump", None)
            memory_max_spatial_noise = getattr(
                opts,
                "front_memory_max_spatial_noise",
                None,
            )
            memory_min_high_active_ratio = getattr(
                opts,
                "front_memory_min_high_active_ratio",
                None,
            )
            if memory_high_threshold is not None:
                front_guidance_kwargs["memory_high_threshold"] = float(
                    memory_high_threshold
                )
            if memory_quality_threshold is not None:
                front_guidance_kwargs["memory_quality_threshold"] = float(
                    memory_quality_threshold
                )
            if memory_noise_threshold is not None:
                front_guidance_kwargs["memory_noise_threshold"] = float(
                    memory_noise_threshold
                )
            if memory_min_alignment is not None:
                front_guidance_kwargs["memory_min_alignment"] = float(
                    memory_min_alignment
                )
            if gate_floor is not None:
                front_guidance_kwargs["gate_floor"] = float(gate_floor)
            if memory_quality_power is not None:
                front_guidance_kwargs["memory_quality_power"] = float(
                    memory_quality_power
                )
            if memory_residual_ratio_max is not None:
                front_guidance_kwargs["memory_residual_ratio_max"] = float(
                    memory_residual_ratio_max
                )
            if memory_query_mask_power is not None:
                front_guidance_kwargs["memory_query_mask_power"] = float(
                    memory_query_mask_power
                )
            if memory_query_mask_floor is not None:
                front_guidance_kwargs["memory_query_mask_floor"] = float(
                    memory_query_mask_floor
                )
            if memory_train_gate_gradient_multiplier is not None:
                front_guidance_kwargs[
                    "memory_train_gate_gradient_multiplier"
                ] = float(memory_train_gate_gradient_multiplier)
            if memory_max_area_std is not None:
                front_guidance_kwargs["memory_max_area_std"] = float(
                    memory_max_area_std
                )
            if memory_min_area_std is not None:
                front_guidance_kwargs["memory_min_area_std"] = float(
                    memory_min_area_std
                )
            if memory_min_valid_frame_ratio is not None:
                front_guidance_kwargs["memory_min_valid_frame_ratio"] = float(
                    memory_min_valid_frame_ratio
                )
            if memory_max_area_jump is not None:
                front_guidance_kwargs["memory_max_area_jump"] = float(
                    memory_max_area_jump
                )
            if memory_max_spatial_noise is not None:
                front_guidance_kwargs["memory_max_spatial_noise"] = float(
                    memory_max_spatial_noise
                )
            if memory_min_high_active_ratio is not None:
                front_guidance_kwargs["memory_min_high_active_ratio"] = float(
                    memory_min_high_active_ratio
                )
            transformer.set_front_guidance(**front_guidance_kwargs)
        if hasattr(transformer, "set_front_condition") and getattr(
            opts, "enable_front_condition", False
        ):
            transformer.set_front_condition(
                enable=True,
                use_boundary=bool(
                    getattr(opts, "front_condition_use_boundary", True)
                ),
                strength_init=float(
                    getattr(opts, "front_condition_strength_init", 1.0)
                ),
                gate_init=float(
                    getattr(opts, "front_condition_gate_init", 0.05)
                ),
                gate_max=float(getattr(opts, "front_condition_gate_max", 0.25)),
                video_scale_init=float(
                    getattr(opts, "front_condition_video_scale_init", 0.0)
                ),
                front_scale_init=float(
                    getattr(opts, "front_condition_front_scale_init", 0.0)
                ),
                boundary_scale_init=float(
                    getattr(opts, "front_condition_boundary_scale_init", 0.0)
                ),
            )
        # transformer = transformer.to(opts.weight_dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae"
        ).to(opts.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype
        )
        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[opts.sampler_name]
        scheduler = Choosen_Scheduler.from_pretrained(
            opts.model_name, subfolder="scheduler"
        )

        self.pipeline = TrajCrafter_Pipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=opts.weight_dtype,
        )

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()

    def run_gradio(self, input_video, stride, radius_scale, pose, steps, seed):
        frames = read_video_frames(
            input_video, self.opts.video_length, stride, self.opts.max_res
        )
        prompt = self.get_caption(self.opts, frames[self.opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            self.opts.near,
            self.opts.far,
            self.opts.depth_inference_steps,
            self.opts.depth_guidance_scale,
            window_size=self.opts.window_size,
            overlap=self.opts.overlap,
        ).to(self.opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0
            - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        num_frames = frames.shape[0]
        assert num_frames == self.opts.video_length
        radius_scale = float(radius_scale)
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(self.opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(self.opts.device)
            .unsqueeze(0)
        )

        # import pdb
        # pdb.set_trace()
        theta, phi, r, x, y = [float(i) for i in pose.split(';')]
        # theta,phi,r,x,y = [float(i) for i in theta.split()],[float(i)
        # for i in phi.split()],[float(i)
        # for i in r.split()],[float(i) for i in x.split()],[float(i) for i in y.split()]
        # target mode
        poses = generate_traj_specified(
            c2w_init, theta, phi, r * radius, x, y, num_frames, self.opts.device,
            direct=getattr(self.opts, 'direct_pose', False),
        )
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[self.opts.anchor_idx : self.opts.anchor_idx + 1].repeat(
            num_frames, 1, 1
        )
        pose_t = poses

        warped_images = []
        masks = []
        for i in tqdm(range(self.opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                self.opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames, cond_video, cond_masks = self._direct_resize_inference_triplet(
            frames, cond_video, cond_masks, self.opts.sample_size
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'input.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = self._build_frames_ref(self.opts, frames[:, :, :10, :, :])
        front_video = self._build_front_video(self.opts)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=self.opts.device).manual_seed(seed)

        # del self.depth_estimater
        # del self.caption_processor
        # del self.captioner
        # gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=self.opts.video_length,
                negative_prompt=self.opts.negative_prompt,
                height=self.opts.sample_size[0],
                width=self.opts.sample_size[1],
                generator=generator,
                guidance_scale=self.opts.diffusion_guidance_scale,
                num_inference_steps=steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                front_video=front_video,
                **self._latent_routing_kwargs(self.opts),
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(self.opts.save_dir, 'gen.mp4'),
            fps=self.opts.fps,
        )

        viz = True
        if viz:
            self._save_viz(
                frames[0],
                sample[0],
                os.path.join(self.opts.save_dir, 'viz.mp4'),
                fps=self.opts.fps * 2,
                device=self.opts.device,
            )
        return os.path.join(self.opts.save_dir, 'viz.mp4')
