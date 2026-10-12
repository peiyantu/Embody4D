# Weights

Large files are released at https://huggingface.co/pytu6/Embody4D.

This open-source tree includes the released project-specific weights: the current best latent expert weights and the LIBERO checkpoint.
The Pi3X checkpoint used by the default warp backend is also available under `Pi3X/` in this Hub repository.

```text
weights/checkpoint/transformer/
weights/latent_residual_weighted/final/
weights/libero_checkpoint/transformer/
```

Base pretrained models are still external and should be downloaded or placed separately:

```text
checkpoints/CogVideoX-Fun-V1.1-5b-InP
checkpoints/DepthCrafter
checkpoints/stable-video-diffusion-img2vid
checkpoints/blip2-opt-2.7b   # external base model
weights/sam3/sam3.pt         # optional SAM3 front-mask checkpoint
```

See `weights/BEST_CONFIG.md` and `weights/best_latent_config.json` for the validated latent route settings. For LIBERO runs, use `CHECKPOINT_DIR=weights/libero_checkpoint`.

