# Embody4D

Embody4D is a video-conditioned 4D view synthesis pipeline with controllable camera motion. This repository contains the project webpage and the released inference code.

The public inference defaults are **Pi3X warping**, **gradual camera motion**, target pose `0 -10 0 0 0`, and **latent experts disabled**. Large project weights are hosted separately at [Hugging Face](https://huggingface.co/pytu6/Embody4D).

## Download only the inference code

The repository also contains the project webpage. The following sparse checkout downloads only `code/`; webpage files are not checked out:

```bash
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/peiyantu/Embody4D.git
cd Embody4D
git sparse-checkout init --cone
git sparse-checkout set code
git checkout
cd code
```

The full inference documentation is in [`code/README.md`](code/README.md).

## Environment

Validated on Linux with an NVIDIA GPU, Python 3.10, CUDA 12.1 compatible PyTorch, and 80 GB of GPU memory. Smaller GPUs may need fewer frames, lower resolution, or fewer diffusion steps.

```bash
conda create -n embody4d python=3.10 -y
conda activate embody4d

# Choose the PyTorch wheel that matches your CUDA driver.
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check the environment:

```bash
python -c "import torch, diffusers, transformers; print(torch.__version__, torch.cuda.is_available(), diffusers.__version__, transformers.__version__)"
```

## Download weights

Authenticate with Hugging Face, then download the released project weights into the code checkout:

```bash
hf auth login
hf download pytu6/Embody4D --local-dir ./weights
```

The released project files include:

```text
weights/
  checkpoint/transformer/                         # default transformer checkpoint
  latent_residual_weighted/final/latent_residual_mse.pt
  libero_checkpoint/transformer/                 # optional LIBERO checkpoint
  Pi3X/model.safetensors                         # default Pi3X warp checkpoint
```

The base checkpoints are separate upstream model releases. Place them under `checkpoints/`, or set `PRETRAINED_ROOT` to another directory:

```text
checkpoints/
  CogVideoX-Fun-V1.1-5b-InP/
  DepthCrafter/
  stable-video-diffusion-img2vid/
  blip2-opt-2.7b/
```

## Quick start

```bash
VIDEO_PATH=/path/to/input.mp4 \
PRETRAINED_ROOT=/path/to/checkpoints \
BLIP_PATH=/path/to/checkpoints/blip2-opt-2.7b \
PI3_CKPT=/path/to/weights/Pi3X/model.safetensors \
bash scripts/infer.sh
```

The wrapper uses 49 frames and 10 fps by default. Outputs are written to `outputs/`.

Useful overrides:

```bash
VIDEO_LENGTH=49
TARGET_POSE='0 -10 0 0 0'
MODE=gradual
WARP_BACKEND=pi3             # use depth for the DepthCrafter backend
DIFFUSION_INFERENCE_STEPS=50
DEPTH_INFERENCE_STEPS=5
```

To use the optional latent expert route, set `ENABLE_LATENT_EXPERTS=1` and follow the released settings in [`code/weights/BEST_CONFIG.md`](code/weights/BEST_CONFIG.md).

## Project structure

```text
code/          Inference implementation, scripts, Pi3X/SAM3 integrations, and metadata
page.html      Project webpage
index.html     GitHub Pages entry point
```

Base model licenses and the optional SAM3 checkpoint are governed by their upstream projects. See [`code/README.md`](code/README.md) for detailed paths, optional front guidance, and troubleshooting.

## Links

- [Project webpage](https://peiyantu.github.io/Embody4D)
- [Inference documentation](code/README.md)
- [Released weights](https://huggingface.co/pytu6/Embody4D)
