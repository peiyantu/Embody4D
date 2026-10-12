from demo import TrajCrafter
import os
from datetime import datetime
import argparse
import torch
import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    stripped = prompt.strip()
    if not stripped:
        return None
    return stripped


def front_mask_quality_stats(
    video_path: str,
    *,
    high_threshold: float = 0.55,
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    areas = []
    high_areas = []
    soft_areas = []
    tv_scores = []
    area_jumps = []
    prev_area = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = frame.astype(np.float32) / 255.0
            area = float(mask.mean())
            areas.append(area)
            high_area = float((mask >= high_threshold).mean())
            soft_area = float(
                ((mask > 0.02) & (mask < high_threshold)).mean()
            )
            high_areas.append(high_area)
            soft_areas.append(soft_area)
            if mask.shape[1] > 1:
                noise_x = float(np.abs(mask[:, 1:] - mask[:, :-1]).mean())
            else:
                noise_x = 0.0
            if mask.shape[0] > 1:
                noise_y = float(np.abs(mask[1:, :] - mask[:-1, :]).mean())
            else:
                noise_y = 0.0
            tv_scores.append(0.5 * (noise_x + noise_y))
            if prev_area is not None:
                area_jumps.append(abs(area - prev_area))
            prev_area = area
    finally:
        cap.release()
    if not areas:
        return {
            "frame_count": 0.0,
            "area_mean": float("inf"),
            "area_std": float("inf"),
            "max_area_jump": float("inf"),
            "valid_frame_ratio": 0.0,
            "spatial_noise": float("inf"),
            "high_active_ratio": 0.0,
        }
    area_array = np.asarray(areas, dtype=np.float32)
    high_array = np.asarray(high_areas, dtype=np.float32)
    soft_array = np.asarray(soft_areas, dtype=np.float32)
    active_array = np.maximum(high_array + soft_array, 1e-8)
    valid = (area_array >= min_area) & (area_array <= max_area)
    return {
        "frame_count": float(len(areas)),
        "area_mean": float(area_array.mean()),
        "area_std": float(area_array.std()),
        "max_area_jump": float(max(area_jumps) if area_jumps else 0.0),
        "valid_frame_ratio": float(valid.mean()),
        "spatial_noise": float(np.asarray(tv_scores, dtype=np.float32).mean()),
        "high_active_ratio": float((high_array / active_array).mean()),
    }


def front_mask_area_std(video_path: str) -> float:
    return front_mask_quality_stats(video_path)["area_std"]


def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, default='./experiments/', help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video_length', type=int, default=49, help='Length of the video frames'
    )
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument('--server_name', type=str, help='Server IP address')

    ## render
    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )
    parser.add_argument('--camera', type=str, default='traj', help='traj or target')
    parser.add_argument(
        '--mode', type=str, default='gradual', help='gradual, bullet or direct'
    )
    parser.add_argument(
        '--mask', action='store_true', default=False, help='Clean the pcd if true'
    )
    parser.add_argument(
        '--use_first_frame_mask',
        action='store_true',
        default=False,
        help='Use first frame mask for all frames in warp; mask stays fixed after creation',
    )
    parser.add_argument(
        '--warp_path',
        type=str,
        default=None,
        help='Optional precomputed render video path (skip internal warp when used with --mask_path)',
    )
    parser.add_argument(
        '--mask_path',
        type=str,
        default=None,
        help='Optional precomputed mask video path (skip internal mask generation when used with --warp_path)',
    )
    parser.add_argument(
        '--warp_backend',
        type=str,
        choices=['depth', 'pi3'],
        default='pi3',
        help='Warp/mask backend: depth uses DepthCrafter+Warper; pi3 uses Pi3X depth with the same target_pose/traj Warper path.',
    )
    parser.add_argument(
        '--pi3_repo_path',
        type=str,
        default=None,
        help='Path to the Pi3 repository. Defaults to PI3_REPO_PATH, then the project-local third_party/Pi3.',
    )
    parser.add_argument(
        '--pi3_target_video',
        type=str,
        default=None,
        help='Deprecated legacy option. The Pi3 backend now ignores this and uses --target_pose/--traj_txt like the depth backend.',
    )
    parser.add_argument(
        '--pi3_output_dir',
        type=str,
        default=None,
        help='Optional directory for Pi3 warped_rgb.mp4 / warped_mask.mp4. Defaults to <save_dir>/pi3_warp.',
    )
    parser.add_argument(
        '--pi3_device',
        type=str,
        default=None,
        help='Device for Pi3 reconstruction. Defaults to --device.',
    )
    parser.add_argument(
        '--pi3_max_frames',
        type=int,
        default=0,
        help='Max frames for Pi3 warp. 0 means use --video_length when positive, otherwise all overlapping frames.',
    )
    parser.add_argument(
        '--pi3_model',
        type=str,
        default=None,
        help='Optional Pi3X Hugging Face model id; exported as PI3X_MODEL for the Pi3 backend.',
    )
    parser.add_argument(
        '--pi3_ckpt',
        type=str,
        default=None,
        help='Optional local Pi3X checkpoint file; exported as PI3X_CKPT for the Pi3 backend.',
    )
    parser.add_argument(
        '--pi3_cache_dir',
        type=str,
        default=None,
        help='Optional Pi3X Hugging Face cache dir; exported as PI3X_CACHE_DIR for the Pi3 backend.',
    )
    parser.add_argument(
        '--pi3_local_files_only',
        action='store_true',
        help='Require Pi3X weights to be available locally; exported as PI3X_LOCAL_FILES_ONLY=1.',
    )
    parser.add_argument(
        '--pi3_reuse_warp',
        action='store_true',
        help='Reuse existing Pi3 warped_rgb.mp4 / warped_mask.mp4 / meta.json under --pi3_output_dir or <save_dir>/pi3_warp.',
    )
    parser.add_argument(
        '--no_pi3_scale_align',
        dest='pi3_enable_scale_align',
        action='store_false',
        help='Deprecated legacy option for the old source-to-target Pi3 path; ignored by the current Pi3 depth Warper backend.',
    )
    parser.set_defaults(pi3_enable_scale_align=True)
    parser.add_argument(
        '--traj_txt',
        type=str,
        help="Required for 'traj' camera, a txt file that specify camera trajectory",
    )
    parser.add_argument(
        '--target_pose',
        type=str,
        default=None,
        help="Required for 'target' mode, 5 numbers: theta phi r x y (e.g. '0 -30 0 0 0'). Use quotes for negative values.",
    )
    parser.add_argument(
        '--direct_pose',
        action='store_true',
        default=False,
        help='Use target pose for all frames (direct to angle, no 0→target interpolation)',
    )
    parser.add_argument(
        '--near', type=float, default=0.0001, help='Near clipping plane distance'
    )
    parser.add_argument(
        '--far', type=float, default=10000.0, help='Far clipping plane distance'
    )
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')

    ## diffusion
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU memory mode',
    )
    # parser.add_argument('--model_name', type=str, default='checkpoints/CogVideoX-Fun-V1.1-5b-InP', help='Path to the model')
    parser.add_argument(
        '--model_name',
        type=str,
        default='alibaba-pai/CogVideoX-Fun-V1.1-5b-InP',
        help='Path to the model',
    )
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )
    # parser.add_argument('--transformer_path', type=str, default='checkpoints/TrajectoryCrafter/crosstransformer', help='Path to the pretrained transformer model')
    parser.add_argument(
        '--transformer_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "weights", "checkpoint", "transformer"),
        help='Path to the pretrained transformer model',
    )
    parser.add_argument(
        '--front_guidance_fallback_transformer_path',
        type=str,
        default=None,
        help=(
            'Optional transformer path used when the foreground-memory '
            'sample-quality gate disables front guidance.'
        ),
    )
    parser.add_argument(
        '--front_guidance_fallback_output_dir',
        type=str,
        default=None,
        help=(
            'Optional output root used when the foreground-memory sample-quality '
            'gate disables front guidance. If <root>/<exp_name>/gen.mp4 exists, '
            'it is linked into this run and inference exits successfully.'
        ),
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[384, 672],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--prompt', type=str, default=None, help='Prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument('--blip_path', type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument(
        '--refvideo',
        type=str,
        default=None,
        help='Optional reference video path; when provided, overrides the default frames_ref sliced from input video',
    )
    parser.add_argument(
        '--frontvideo',
        type=str,
        default=None,
        help='Optional front mask video path used for front guidance or front condition.',
    )
    parser.add_argument(
        '--enable_front_guidance',
        action='store_true',
        help='Enable front-guided cross attention in inference.',
    )
    parser.add_argument(
        '--sam3_front_mask',
        action='store_true',
        help='Generate --frontvideo from --video_path with SAM3 before inference when --frontvideo is not provided.',
    )
    parser.add_argument(
        '--sam3_repo_path',
        type=str,
        default=None,
        help='Path to the SAM3 repository. Defaults to the project-local third_party/sam3.',
    )
    parser.add_argument(
        '--sam3_python_bin',
        type=str,
        default=None,
        help='Python executable used for SAM3 generation. Defaults to the sam3 conda env when available.',
    )
    parser.add_argument(
        '--sam3_checkpoint_path',
        type=str,
        default=None,
        help='Local SAM3 checkpoint path. Defaults to the project-local weights/sam3/sam3.pt.',
    )
    parser.add_argument(
        '--sam3_output_dir',
        type=str,
        default=None,
        help='Optional directory for generated SAM3 front masks. Defaults to <save_dir>/sam3_front_mask.',
    )
    parser.add_argument(
        '--sam3_prompts',
        nargs='+',
        default=['object', 'robot arm', 'moving object'],
        help='SAM3 text prompts used to generate the front mask.',
    )
    parser.add_argument(
        '--sam3_prompt_frame_index',
        type=int,
        default=0,
        help='Frame index where SAM3 text prompts are added.',
    )
    parser.add_argument(
        '--sam3_gpus',
        type=str,
        default=None,
        help='CUDA_VISIBLE_DEVICES value for SAM3 generation. Defaults to the current inference device.',
    )
    parser.add_argument(
        '--sam3_reuse_mask',
        action='store_true',
        help='Reuse an existing generated SAM3 mask under --sam3_output_dir or <save_dir>/sam3_front_mask.',
    )
    parser.add_argument(
        '--sam3_force_mask',
        action='store_true',
        help='Regenerate the SAM3 mask even if the expected output already exists.',
    )
    parser.add_argument(
        '--front_guidance_target_scale',
        type=float,
        default=0.6,
        help='Target blend scale for front-guided attention.',
    )
    parser.add_argument(
        '--front_guidance_boost_strength',
        type=float,
        default=5.0,
        help='Bias strength added by front mask on cross-attention keys.',
    )
    parser.add_argument(
        '--front_guidance_mode',
        type=str,
        default='blend',
        choices=['blend', 'soft_bias', 'residual_gate', 'foreground_memory'],
        help=(
            'Front guidance mode: legacy blend, zero-mean soft_bias, or '
            'residual_gate value residual, or foreground_memory branch.'
        ),
    )
    parser.add_argument(
        '--front_guidance_min_area',
        type=float,
        default=0.0,
        help='Minimum valid front-mask area for soft_bias.',
    )
    parser.add_argument(
        '--front_guidance_max_area',
        type=float,
        default=1.0,
        help='Maximum valid front-mask area for soft_bias.',
    )
    parser.add_argument(
        '--front_guidance_gate_floor',
        type=float,
        default=None,
        help='Optional minimum foreground-memory gate value before quality gating.',
    )
    parser.add_argument(
        '--front_memory_high_threshold',
        type=float,
        default=None,
        help='Optional high-confidence mask threshold override for foreground_memory.',
    )
    parser.add_argument(
        '--front_memory_quality_threshold',
        type=float,
        default=None,
        help='Optional sample quality threshold override for foreground_memory.',
    )
    parser.add_argument(
        '--front_memory_noise_threshold',
        type=float,
        default=None,
        help='Optional mask total-variation noise threshold override for foreground_memory.',
    )
    parser.add_argument(
        '--front_memory_min_alignment',
        type=float,
        default=None,
        help='Optional raw memory/global attention alignment threshold for foreground_memory.',
    )
    parser.add_argument(
        '--front_memory_quality_power',
        type=float,
        default=None,
        help='Optional exponent for foreground-memory quality gating.',
    )
    parser.add_argument(
        '--front_memory_residual_ratio_max',
        type=float,
        default=None,
        help='Optional cap on foreground-memory residual norm relative to global attention.',
    )
    parser.add_argument(
        '--front_memory_query_mask_power',
        type=float,
        default=None,
        help=(
            'Optional exponent for query-side front-mask gating of foreground-memory '
            'residuals. Values <=0 disable this local gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_query_mask_floor',
        type=float,
        default=None,
        help=(
            'Optional floor added to query-side front-mask gating of foreground-memory '
            'residuals. Use 0 for strict foreground-only gating.'
        ),
    )
    parser.add_argument(
        '--front_memory_train_gate_gradient_multiplier',
        type=float,
        default=None,
        help=(
            'Training-only straight-through gate-gradient multiplier stored on '
            'foreground_memory checkpoints. It has no effect during inference.'
        ),
    )
    parser.add_argument(
        '--front_memory_max_area_std',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance for this sample when the '
            'front-mask area std over time is larger than this value. Negative '
            'values disable this gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_min_area_std',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance for this sample when the '
            'front-mask area std over time is smaller than this value. Negative '
            'values disable this gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_min_valid_frame_ratio',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance when too few mask frames '
            'have area inside the configured front guidance area range. Values '
            '<=0 disable this gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_max_area_jump',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance when the maximum '
            'frame-to-frame mask area jump is larger than this value. Negative '
            'values disable this gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_max_spatial_noise',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance when the average spatial '
            'mask total variation is larger than this value. Negative values '
            'disable this gate.'
        ),
    )
    parser.add_argument(
        '--front_memory_min_high_active_ratio',
        type=float,
        default=None,
        help=(
            'Disable foreground-memory front guidance when high-confidence '
            'pixels are too small a fraction of active mask pixels. Values <=0 '
            'disable this gate.'
        ),
    )
    parser.add_argument(
        '--enable_front_condition',
        action='store_true',
        help='Enable front mask conditioning in the transformer.',
    )
    parser.add_argument(
        '--front_condition_strength_init',
        type=float,
        default=1.0,
        help='Initial strength for front mask conditioning.',
    )
    parser.add_argument(
        '--front_condition_gate_init',
        type=float,
        default=0.05,
        help='Initial gate value for front mask conditioning.',
    )
    parser.add_argument(
        '--front_condition_gate_max',
        type=float,
        default=0.25,
        help='Maximum gate value for front mask conditioning.',
    )
    parser.add_argument(
        '--front_condition_video_scale_init',
        type=float,
        default=0.0,
        help='Initial video scale for front mask conditioning.',
    )
    parser.add_argument(
        '--front_condition_front_scale_init',
        type=float,
        default=0.0,
        help='Initial front mask scale for front mask conditioning.',
    )
    parser.add_argument(
        '--front_condition_boundary_scale_init',
        type=float,
        default=0.0,
        help='Initial boundary scale for front mask conditioning.',
    )
    parser.add_argument(
        '--no_front_condition_boundary',
        dest='front_condition_use_boundary',
        action='store_false',
        help='Disable front mask boundary input for front condition.',
    )
    parser.set_defaults(front_condition_use_boundary=True)
    parser.add_argument(
        '--front_loss_weighted_model',
        action='store_true',
        help='Mark this checkpoint as trained with front loss reweighting; inference stays no-front.',
    )
    parser.add_argument(
        '--latent_residual_checkpoint',
        type=str,
        default=None,
        help='Optional latent residual model used only when the transformer checkpoint contains FFN expert routers.',
    )
    parser.add_argument(
        '--latent_route_top_k',
        type=int,
        default=2,
        help='Number of latent FFN expert routes to blend per token.',
    )
    parser.add_argument(
        '--latent_route_secondary_weight',
        type=float,
        default=0.2,
        help='Neighbor expert weight when latent_route_top_k is 2.',
    )
    parser.add_argument(
        '--latent_route_copy_valid_threshold',
        type=float,
        default=0.85,
        help='Soft-valid threshold for copy expert routing.',
    )
    parser.add_argument(
        '--latent_route_copy_delta_threshold',
        type=float,
        default=0.35,
        help='Delta magnitude threshold for copy expert routing.',
    )
    parser.add_argument(
        '--latent_route_inpaint_delta_threshold',
        type=float,
        default=0.50,
        help='Delta magnitude threshold for inpaint expert routing.',
    )
    parser.add_argument(
        '--latent_route_inpaint_valid_threshold',
        type=float,
        default=0.45,
        help='In non-legacy routing, normal inpaint requires soft-valid below this threshold.',
    )
    parser.add_argument(
        '--latent_route_force_inpaint_delta_threshold',
        type=float,
        default=0.85,
        help='In non-legacy routing, force inpaint above this residual magnitude.',
    )
    parser.add_argument(
        '--latent_route_postprocess_mode',
        type=str,
        choices=['legacy', 'hard', 'top2_vote'],
        default='legacy',
        help='Latent route post-processing mode used by FFN expert routing.',
    )
    parser.add_argument(
        '--latent_route_smooth_spatial_kernel',
        type=int,
        default=1,
        help='Odd spatial smoothing kernel for residual magnitude in non-legacy routing.',
    )
    parser.add_argument(
        '--latent_route_smooth_temporal_kernel',
        type=int,
        default=1,
        help='Odd temporal smoothing kernel for residual magnitude and valid mask.',
    )
    parser.add_argument(
        '--latent_route_valid_smooth_spatial_kernel',
        type=int,
        default=1,
        help='Odd spatial smoothing kernel for soft-valid mask in non-legacy routing.',
    )
    parser.add_argument(
        '--latent_route_vote_spatial_kernel',
        type=int,
        default=1,
        help='Odd spatial smoothing kernel for top-2 route weights.',
    )
    parser.add_argument(
        '--latent_route_vote_temporal_kernel',
        type=int,
        default=1,
        help='Odd temporal smoothing kernel for top-2 route weights.',
    )
    parser.add_argument(
        '--latent_route_postprocess_on_video_grid',
        action='store_true',
        help='Upsample route maps to video resolution for smoothing before token pooling.',
    )
    parser.add_argument(
        '--latent_ffn_expert_scale',
        type=float,
        default=1.0,
        help='Scale applied to FFN expert residuals during inference.',
    )
    parser.add_argument(
        '--latent_route_copy_residual_scale',
        type=float,
        default=1.0,
        help=(
            'Per-token residual scale for copy-routed latent FFN experts. '
            'Use 0.0 to make copy tokens a no-op while repair/inpaint stay active.'
        ),
    )

    ## depth
    # parser.add_argument('--unet_path', type=str, default='checkpoints/DepthCrafter', help='Path to the UNet model')
    parser.add_argument(
        '--unet_path',
        type=str,
        default="tencent/DepthCrafter",
        help='Path to the UNet model',
    )

    # parser.add_argument('--pre_train_path', type=str, default='checkpoints/stable-video-diffusion-img2vid-xt', help='Path to the pre-trained model')
    parser.add_argument(
        '--pre_train_path',
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help='Path to the pre-trained model',
    )
    parser.add_argument(
        '--cpu_offload', type=str, default='model', help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    has_warp_path = bool(opts.warp_path)
    has_mask_path = bool(opts.mask_path)
    if has_warp_path != has_mask_path:
        raise ValueError(
            "Please provide both --warp_path and --mask_path together, or provide neither."
        )
    if has_warp_path:
        if not os.path.isfile(opts.warp_path):
            raise FileNotFoundError(f"warp_path not found: {opts.warp_path}")
        if not os.path.isfile(opts.mask_path):
            raise FileNotFoundError(f"mask_path not found: {opts.mask_path}")
    if opts.target_pose is not None:
        opts.target_pose = [float(x) for x in opts.target_pose.split()]
        assert len(opts.target_pose) == 5, "target_pose must be 5 numbers: theta phi r x y"
    opts.prompt = normalize_prompt(opts.prompt)

    opts.weight_dtype = torch.bfloat16
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = (
            f'{prefix}_{os.path.splitext(os.path.basename(opts.video_path))[0]}'
        )
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)

    if opts.front_loss_weighted_model:
        if opts.enable_front_guidance or opts.enable_front_condition:
            raise ValueError(
                "--front_loss_weighted_model is no-front inference and cannot be "
                "combined with --enable_front_guidance or --enable_front_condition."
            )
        if opts.frontvideo:
            raise ValueError(
                "--front_loss_weighted_model does not take --frontvideo at inference time."
            )
    if opts.sam3_front_mask and not opts.frontvideo:
        from sam3_mask_backend import generate_sam3_front_mask

        opts.frontvideo = generate_sam3_front_mask(opts)
        print(f"SAM3 front mask generated: {opts.frontvideo}")
    if (
        (opts.enable_front_guidance or opts.enable_front_condition)
        and not opts.frontvideo
    ):
        raise ValueError(
            "Front guidance/condition requires --frontvideo to be provided."
        )
    if opts.frontvideo and not os.path.isfile(opts.frontvideo):
        raise FileNotFoundError(f"frontvideo not found: {opts.frontvideo}")
    if opts.enable_front_guidance and opts.front_guidance_mode == "foreground_memory":
        sample_gate_enabled = any(
            (
                opts.front_memory_max_area_std is not None
                and opts.front_memory_max_area_std >= 0.0,
                opts.front_memory_min_area_std is not None
                and opts.front_memory_min_area_std >= 0.0,
                opts.front_memory_min_valid_frame_ratio is not None
                and opts.front_memory_min_valid_frame_ratio > 0.0,
                opts.front_memory_max_area_jump is not None
                and opts.front_memory_max_area_jump >= 0.0,
                opts.front_memory_max_spatial_noise is not None
                and opts.front_memory_max_spatial_noise >= 0.0,
                opts.front_memory_min_high_active_ratio is not None
                and opts.front_memory_min_high_active_ratio > 0.0,
            )
        )
        if sample_gate_enabled:
            high_threshold = (
                opts.front_memory_high_threshold
                if opts.front_memory_high_threshold is not None
                else 0.55
            )
            stats = front_mask_quality_stats(
                opts.frontvideo,
                high_threshold=float(high_threshold),
                min_area=float(opts.front_guidance_min_area),
                max_area=float(opts.front_guidance_max_area),
            )
            reasons = []
            if (
                opts.front_memory_max_area_std is not None
                and opts.front_memory_max_area_std >= 0.0
                and stats["area_std"] > opts.front_memory_max_area_std
            ):
                reasons.append(
                    f"area_std>{opts.front_memory_max_area_std:.6f}"
                )
            if (
                opts.front_memory_min_area_std is not None
                and opts.front_memory_min_area_std >= 0.0
                and stats["area_std"] < opts.front_memory_min_area_std
            ):
                reasons.append(
                    f"area_std<{opts.front_memory_min_area_std:.6f}"
                )
            if (
                opts.front_memory_min_valid_frame_ratio is not None
                and opts.front_memory_min_valid_frame_ratio > 0.0
                and stats["valid_frame_ratio"]
                < opts.front_memory_min_valid_frame_ratio
            ):
                reasons.append(
                    "valid_frame_ratio"
                    f"<{opts.front_memory_min_valid_frame_ratio:.6f}"
                )
            if (
                opts.front_memory_max_area_jump is not None
                and opts.front_memory_max_area_jump >= 0.0
                and stats["max_area_jump"] > opts.front_memory_max_area_jump
            ):
                reasons.append(
                    f"max_area_jump>{opts.front_memory_max_area_jump:.6f}"
                )
            if (
                opts.front_memory_max_spatial_noise is not None
                and opts.front_memory_max_spatial_noise >= 0.0
                and stats["spatial_noise"] > opts.front_memory_max_spatial_noise
            ):
                reasons.append(
                    "spatial_noise"
                    f">{opts.front_memory_max_spatial_noise:.6f}"
                )
            if (
                opts.front_memory_min_high_active_ratio is not None
                and opts.front_memory_min_high_active_ratio > 0.0
                and stats["high_active_ratio"]
                < opts.front_memory_min_high_active_ratio
            ):
                reasons.append(
                    "high_active_ratio"
                    f"<{opts.front_memory_min_high_active_ratio:.6f}"
                )
            status = "disabled" if reasons else "kept"
            print(
                f"Front guidance {status} by sample-quality gate: "
                f"frames={stats['frame_count']:.0f}, "
                f"area_mean={stats['area_mean']:.6f}, "
                f"area_std={stats['area_std']:.6f}, "
                f"max_area_jump={stats['max_area_jump']:.6f}, "
                f"valid_frame_ratio={stats['valid_frame_ratio']:.6f}, "
                f"spatial_noise={stats['spatial_noise']:.6f}, "
                f"high_active_ratio={stats['high_active_ratio']:.6f}"
                + (f", reasons={';'.join(reasons)}" if reasons else "")
            )
            if reasons:
                opts.enable_front_guidance = False
                fallback_output_dir = getattr(
                    opts,
                    "front_guidance_fallback_output_dir",
                    None,
                )
                if fallback_output_dir:
                    fallback_save_dir = os.path.join(
                        fallback_output_dir,
                        opts.exp_name,
                    )
                    fallback_gen = os.path.join(fallback_save_dir, "gen.mp4")
                    if not os.path.isfile(fallback_gen):
                        raise FileNotFoundError(
                            "front_guidance_fallback_output_dir gen.mp4 not found: "
                            f"{fallback_gen}"
                        )
                    for filename in ("gen.mp4", "viz.mp4", "input.mp4", "render.mp4", "mask.mp4"):
                        source = os.path.join(fallback_save_dir, filename)
                        if not os.path.exists(source):
                            continue
                        target = os.path.join(opts.save_dir, filename)
                        if os.path.lexists(target):
                            os.remove(target)
                        os.symlink(os.path.realpath(source), target)
                    print(
                        "Front guidance fallback output selected: "
                        f"{fallback_save_dir}"
                    )
                    raise SystemExit(0)
                fallback_transformer_path = getattr(
                    opts,
                    "front_guidance_fallback_transformer_path",
                    None,
                )
                if fallback_transformer_path:
                    if not os.path.exists(fallback_transformer_path):
                        raise FileNotFoundError(
                            "front_guidance_fallback_transformer_path not found: "
                            f"{fallback_transformer_path}"
                        )
                    opts.transformer_path = fallback_transformer_path
                    print(
                        "Front guidance fallback transformer selected: "
                        f"{fallback_transformer_path}"
                    )
    if opts.latent_route_top_k not in (1, 2):
        raise ValueError("--latent_route_top_k currently supports 1 or 2.")
    if not 0.0 <= opts.latent_route_secondary_weight < 1.0:
        raise ValueError("--latent_route_secondary_weight must be in [0, 1).")
    if not 0.0 <= opts.latent_route_copy_residual_scale <= 1.0:
        raise ValueError("--latent_route_copy_residual_scale must be in [0, 1].")
    for name in (
        'latent_route_smooth_spatial_kernel',
        'latent_route_smooth_temporal_kernel',
        'latent_route_valid_smooth_spatial_kernel',
        'latent_route_vote_spatial_kernel',
        'latent_route_vote_temporal_kernel',
    ):
        value = int(getattr(opts, name))
        if value < 1 or value % 2 == 0:
            raise ValueError(f"--{name} must be a positive odd integer.")
    if opts.latent_residual_checkpoint:
        latent_residual_candidates = [opts.latent_residual_checkpoint]
        if not os.path.isabs(opts.latent_residual_checkpoint):
            latent_residual_candidates.append(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), opts.latent_residual_checkpoint)
            )
        if not any(
            os.path.isfile(candidate)
            or os.path.isfile(os.path.join(candidate, "latent_residual_mse.pt"))
            or os.path.isdir(candidate)
            for candidate in latent_residual_candidates
        ):
            raise FileNotFoundError(
                "latent_residual_checkpoint not found: "
                f"{opts.latent_residual_checkpoint}"
            )

    if opts.warp_backend == 'pi3' and not (opts.warp_path and opts.mask_path):
        from pi3_warp_backend import generate_pi3_warp

        opts.warp_path, opts.mask_path, pi3_frames_used = generate_pi3_warp(opts)
        if pi3_frames_used > 0:
            opts.video_length = pi3_frames_used
        print(
            'Pi3 warp backend generated external warp/mask: '
            f'warp_path={opts.warp_path}, mask_path={opts.mask_path}, '
            f'frames={opts.video_length}'
        )

    pvd = TrajCrafter(opts)
    if opts.mode == 'gradual':
        pvd.infer_gradual(opts)
    elif opts.mode == 'direct':
        pvd.infer_direct(opts)
    elif opts.mode == 'bullet':
        pvd.infer_bullet(opts)
    elif opts.mode == 'zoom':
        pvd.infer_zoom(opts)
