# Best Released Project Weights

This directory contains the current best single-checkpoint latent expert release used by the open-source code, plus the released LIBERO checkpoint.

## Included Weights

```text
weights/checkpoint/transformer/
  config.json
  diffusion_pytorch_model-00001-of-00002.safetensors
  diffusion_pytorch_model-00002-of-00002.safetensors
  diffusion_pytorch_model.safetensors.index.json

weights/latent_residual_weighted/final/
  config.json
  latent_residual_mse.pt

weights/libero_checkpoint/transformer/
  config.json
  diffusion_pytorch_model.safetensors
```

## Best Inference Settings

Use these settings for the best validated latent route configuration:

```bash
ENABLE_LATENT_EXPERTS=1
LATENT_RESIDUAL_CHECKPOINT=weights/latent_residual_weighted/final
CHECKPOINT_DIR=weights/checkpoint
LATENT_ROUTE_TOP_K=2
LATENT_ROUTE_SECONDARY_WEIGHT=0.10
LATENT_ROUTE_POSTPROCESS_MODE=top2_vote
LATENT_ROUTE_VOTE_SPATIAL_KERNEL=3
LATENT_ROUTE_VOTE_TEMPORAL_KERNEL=3
LATENT_FFN_EXPERT_SCALE=0.025
LATENT_ROUTE_COPY_RESIDUAL_SCALE=0.0
```

The release checkpoint corresponds to the internally selected `checkpoint-5000` latent expert. Later checkpoints tested at the same scale showed visible view/layout drift on quick samples, so they are not included.

## LIBERO Checkpoint

For LIBERO experiments, use the included task-specific checkpoint root:

```bash
CHECKPOINT_DIR=weights/libero_checkpoint
```

## VBench Reference

Best single latent route candidate, mixed only where it improved samples:

```text
subject_consistency     +0.00001007
background_consistency  +0.00005717
motion_smoothness       +0.00000030
dynamic_degree           0.00000000
aesthetic_quality       +0.00011988
imaging_quality         +0.00023972
simple_avg              +0.00007119
```

A more aggressive per-video route using multiple historical candidates reached a higher simple average, but the released single-checkpoint latent expert is the cleanest reproducible weight package.
