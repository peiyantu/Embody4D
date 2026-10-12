# Weights

Large files are released at https://huggingface.co/pytu6/Embody4D.

This open-source tree includes the released project-specific weights: the current best latent expert weights and the LIBERO checkpoint.

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
weights/blip2-opt-2.7b
weights/sam3/sam3.pt   # only needed for SAM3 front-mask generation
```

See `weights/BEST_CONFIG.md` and `weights/best_latent_config.json` for the validated latent route settings. For LIBERO runs, use `CHECKPOINT_DIR=weights/libero_checkpoint`.

