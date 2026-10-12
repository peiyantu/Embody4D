# Embody4D Inference Code

This directory contains the released inference pipeline for Embody4D: video-conditioned 4D view synthesis with controllable camera motion.

The default command uses **Pi3X warping**, **gradual camera motion**, a target pose of `0 -10 0 0 0`, and keeps the optional latent experts **disabled**. The released latent expert weights can be enabled when you need the best latent-route configuration.

## Repository layout

```text
src/                 Inference entry point and model implementation
scripts/             Reproducible shell entry points
third_party/Pi3/     Pi3X source used by the default warp backend
third_party/sam3/    Optional SAM3 source for front-mask guidance
assets/              Small trajectories and example assets
checkpoints/         Placeholder for external base checkpoints
weights/              Placeholder for released project weights
requirements.txt     Python dependencies for the unified environment
```

Large model files are not stored in GitHub. Download the project weights from the [Embody4D Hugging Face repository](https://huggingface.co/peiyantu/Embody4D), then place them in the paths described below.

## Hardware and software

The released pipeline was validated on Linux with an NVIDIA GPU, Python 3.10, CUDA 12.1 compatible PyTorch, and 80 GB of GPU memory. Smaller GPUs may require fewer frames, lower resolution, or fewer diffusion steps.

The commands below create a clean environment. Install PyTorch first so that the wheel matches the CUDA driver on your machine:

```bash
conda create -n embody4d python=3.10 -y
conda activate embody4d

# Example for CUDA 12.1. Choose the PyTorch index that matches your driver.
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check the installation before downloading the models:

```bash
python - <<'PY'
import torch
import diffusers
import transformers
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
PY
```

The Pi3X code is already included in `third_party/Pi3`. Do not install its standalone `requirements.txt` on top of this environment because it pins a different PyTorch and NumPy version.

## Model files

### Project weights

Download the released files from [Hugging Face](https://huggingface.co/peiyantu/Embody4D). The default layout is:

```text
weights/
  checkpoint/transformer/                         # released transformer shards
  latent_residual_weighted/final/latent_residual_mse.pt
  libero_checkpoint/transformer/                 # optional LIBERO checkpoint
  blip2-opt-2.7b/                                # image captioning model
  Pi3X/model.safetensors                         # Pi3X checkpoint
```

`weights/checkpoint/` and `weights/latent_residual_weighted/final/` are sufficient for the default project checkpoint plus optional latent experts. The `blip2-opt-2.7b` and Pi3X files are required by the default captioning and warp stages.

### Base checkpoints

Download the base models used by the pipeline from their upstream releases and place them under `checkpoints/` (or point the command to another directory with `PRETRAINED_ROOT`):

```text
checkpoints/
  CogVideoX-Fun-V1.1-5b-InP/
  DepthCrafter/
  stable-video-diffusion-img2vid/
```

The `DepthCrafter` checkpoint is only needed when `WARP_BACKEND=depth`. Private datasets and training artifacts are not part of this release.

## Quick start

Set paths for the downloaded models and run one video:

```bash
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
BLIP_PATH=/path/to/weights/blip2-opt-2.7b \
PI3_CKPT=/path/to/weights/Pi3X/model.safetensors \
bash scripts/infer.sh
```

The wrapper defaults to 49 frames and 10 fps. It writes the generated video and intermediate files under `outputs/`. `scripts/infer_gradual.sh` is an explicit alias for the same gradual mode:

```bash
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
PI3_CKPT=/path/to/weights/Pi3X/model.safetensors \
bash scripts/infer_gradual.sh
```

Useful overrides:

```bash
VIDEO_LENGTH=49                 # number of frames
TARGET_POSE='0 -10 0 0 0'        # yaw, pitch, roll, x, y
MODE=gradual                     # gradual, direct, bullet, or zoom
WARP_BACKEND=pi3                 # pi3 (default) or depth
DIFFUSION_INFERENCE_STEPS=50
DEPTH_INFERENCE_STEPS=5
```

The script passes additional arguments through to `src/inference.py`, so you can also append flags such as `--prompt` or `--exp_name`.

To use the original DepthCrafter warp backend:

```bash
WARP_BACKEND=depth \
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
bash scripts/infer.sh
```

## Optional latent experts

Latent experts are off by default. Enable them only after downloading the latent residual file:

```bash
ENABLE_LATENT_EXPERTS=1 \
LATENT_RESIDUAL_CHECKPOINT=weights/latent_residual_weighted/final \
LATENT_ROUTE_POSTPROCESS_MODE=top2_vote \
LATENT_ROUTE_TOP_K=2 \
LATENT_ROUTE_SECONDARY_WEIGHT=0.10 \
LATENT_ROUTE_VOTE_SPATIAL_KERNEL=3 \
LATENT_ROUTE_VOTE_TEMPORAL_KERNEL=3 \
LATENT_FFN_EXPERT_SCALE=0.025 \
LATENT_ROUTE_COPY_RESIDUAL_SCALE=0.0 \
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
PI3_CKPT=/path/to/weights/Pi3X/model.safetensors \
bash scripts/infer.sh
```

For the corresponding configuration notes, see [`weights/BEST_CONFIG.md`](weights/BEST_CONFIG.md). For LIBERO experiments, set `CHECKPOINT_DIR=weights/libero_checkpoint`.

## Optional front guidance

Front guidance is disabled by default. To use the included SAM3 integration, provide a SAM3 checkpoint and enable it explicitly:

```bash
ENABLE_FRONT_GUIDANCE=1 \
USE_SAM3_FRONT_MASK=1 \
SAM3_CHECKPOINT_PATH=weights/sam3/sam3.pt \
VIDEO_PATH=/path/to/input.mp4 \
bash scripts/infer.sh
```

## Reproducibility notes

- The default public configuration is Pi3X + gradual mode + target pose `0 -10 0 0 0`.
- Set `WARP_BACKEND=depth` to use the DepthCrafter warp implementation.
- Set `ENABLE_LATENT_EXPERTS=1` to opt into the latent expert route; leaving it unset keeps the route off.
- The public tree contains source code and lightweight metadata only. Large weights, datasets, logs, and generated videos are intentionally excluded.
- Check each upstream model's license before redistributing its checkpoint.

## Citation

If you use Embody4D in your work, please cite the project paper and the release page at [peiyantu.github.io/Embody4D](https://peiyantu.github.io/Embody4D).
