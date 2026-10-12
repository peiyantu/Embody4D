# Embody4D Open-Source Code + Released Project Weights

This is a cleaned open-source tree for Embody4D experiments. It contains the inference code, scripts, small example assets, lightweight third-party integration code, and the current best released project-specific latent expert weights plus the LIBERO checkpoint.

Large base pretrained models and private datasets are intentionally not included.

## Released Weights

The large project weights are hosted separately on Hugging Face:

```text
https://huggingface.co/peiyantu/Embody4D
```

Download the project weights and Pi3X checkpoint before running the default Pi3 gradual pipeline.

## What Is Included

```text
src/                 Core inference model, routing, depth, mask, and warp helpers
scripts/             Shell entry points for inference and optional Pi3/SAM3 setup
assets/              Small example videos and trajectory text files
third_party/         Lightweight Pi3 and SAM3 source snapshots, with heavy examples removed
weights/checkpoint/transformer/
weights/latent_residual_weighted/final/
weights/libero_checkpoint/transformer/
weights/BEST_CONFIG.md
checkpoints/README.md
```

## External Files You Still Need

Place base pretrained models under `checkpoints/`, or point scripts to your own paths with environment variables:

```text
checkpoints/CogVideoX-Fun-V1.1-5b-InP
checkpoints/DepthCrafter
checkpoints/stable-video-diffusion-img2vid
weights/blip2-opt-2.7b
```

SAM3 is optional and only needed when using automatic front-mask generation:

```text
weights/sam3/sam3.pt
```

## Best Released Latent Configuration

The released project weights correspond to the selected `checkpoint-5000` latent expert. Latent experts are optional and disabled by default. Enable them with:

```bash
ENABLE_LATENT_EXPERTS=1 \
CHECKPOINT_DIR=weights/checkpoint \
LATENT_RESIDUAL_CHECKPOINT=weights/latent_residual_weighted/final \
LATENT_ROUTE_POSTPROCESS_MODE=top2_vote \
LATENT_ROUTE_TOP_K=2 \
LATENT_ROUTE_SECONDARY_WEIGHT=0.10 \
LATENT_ROUTE_VOTE_SPATIAL_KERNEL=3 \
LATENT_ROUTE_VOTE_TEMPORAL_KERNEL=3 \
LATENT_FFN_EXPERT_SCALE=0.025 \
LATENT_ROUTE_COPY_RESIDUAL_SCALE=0.0
```

See `weights/BEST_CONFIG.md` for metric notes and rationale.

For LIBERO experiments, switch the transformer checkpoint root:

```bash
CHECKPOINT_DIR=weights/libero_checkpoint
```

## Inference

Single-video inference:

```bash
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
BLIP_PATH=/path/to/blip2-opt-2.7b \
PI3_CKPT=/path/to/Pi3X/model.safetensors \
bash scripts/infer.sh
```

Front guidance is opt-in. Because front masks can be noisy, keep strong gating and validate a few generated samples before large VBench runs.

## Release Notes

This tree was created from an internal experiment directory by removing private datasets, experiment outputs, logs, backup files, and base pretrained checkpoints. Large project-specific weights are released as external Hugging Face artifacts.

The default inference entry uses Pi3, gradual mode, and target pose `0 -10 0 0 0`; set `WARP_BACKEND=depth` to use DepthCrafter instead. Large weights are external Hugging Face artifacts.
