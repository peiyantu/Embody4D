# pytu 运行手册与测试报告

日期：2026-10-12（Asia/Shanghai）
服务器：pytu（8 × NVIDIA A100 80GB）
项目目录：/home/dataset-assist-0/pytu/Embody4D_open_source

## 已整理内容

- \`src/\`、\`scripts/\`、\`third_party/\`、\`assets/\`、\`data/\` 保持开源代码和小型示例资源。
- 项目专用权重保留在 \`weights/checkpoint/transformer/\`、\`weights/latent_residual_weighted/final/\`、\`weights/libero_checkpoint/transformer/\`。
- 基础模型继续放在服务器共享目录 \`/home/dataset-assist-0/pytu/checkpoints/\`，避免复制 100GB 以上的外部模型。
- \`scripts/infer.sh\` 已支持在切换到 \`src/\` 前解析项目内的相对 checkpoint、router、轨迹和本地模型路径；当环境里没有 \`python3\` 但有 \`python3.10\` 时会自动选择可用解释器。

## pytu 环境

\`\`\`bash
PYTHON_BIN=/home/dataset-assist-0/pytu/miniconda/envs/embody4d/bin/python3.10
PRETRAINED_ROOT=/home/dataset-assist-0/pytu/checkpoints
MODEL_NAME=/home/dataset-assist-0/pytu/checkpoints/CogVideoX-Fun-V1.1-5b-InP
BLIP_PATH=/home/dataset-assist-0/pytu/checkpoints/blip2-opt-2.7b
UNET_PATH=/home/dataset-assist-0/pytu/checkpoints/DepthCrafter
PRE_TRAIN_PATH=/home/dataset-assist-0/pytu/checkpoints/stable-video-diffusion-img2vid
\`\`\`

## 标准推理命令

\`\`\`bash
cd /home/dataset-assist-0/pytu/Embody4D_open_source

PYTHON_BIN=/home/dataset-assist-0/pytu/miniconda/envs/embody4d/bin/python3.10 \
VIDEO_PATH=assets/example/1.mp4 \
PRETRAINED_ROOT=/home/dataset-assist-0/pytu/checkpoints \
MODEL_NAME=/home/dataset-assist-0/pytu/checkpoints/CogVideoX-Fun-V1.1-5b-InP \
BLIP_PATH=/home/dataset-assist-0/pytu/checkpoints/blip2-opt-2.7b \
UNET_PATH=/home/dataset-assist-0/pytu/checkpoints/DepthCrafter \
PRE_TRAIN_PATH=/home/dataset-assist-0/pytu/checkpoints/stable-video-diffusion-img2vid \
ENABLE_LATENT_EXPERTS=1 \
CHECKPOINT_DIR=/home/dataset-assist-0/pytu/Embody4D_open_source/weights/checkpoint \
LATENT_RESIDUAL_CHECKPOINT=/home/dataset-assist-0/pytu/Embody4D_open_source/weights/latent_residual_weighted/final \
LATENT_ROUTE_POSTPROCESS_MODE=top2_vote \
LATENT_ROUTE_TOP_K=2 \
LATENT_ROUTE_SECONDARY_WEIGHT=0.10 \
LATENT_ROUTE_VOTE_SPATIAL_KERNEL=3 \
LATENT_ROUTE_VOTE_TEMPORAL_KERNEL=3 \
LATENT_FFN_EXPERT_SCALE=0.025 \
LATENT_ROUTE_COPY_RESIDUAL_SCALE=0.0 \
bash scripts/infer_direct.sh \
  --sample_size 384 672 \
  --diffusion_inference_steps 50 \
  --depth_inference_steps 5
\`\`\`

## 测试结果

### 静态检查

- 8 个 shell 入口均通过 \`bash -n\`。
- \`src/\`、\`third_party/Pi3\`、\`third_party/sam3\` 通过 Python 3.10 \`compileall\`。
- 核心依赖导入成功：PyTorch 2.4.0+cu121、Diffusers 0.32.1、Transformers 4.47.0、CUDA 可用且检测到 8 张 GPU。

### 实际管线

- 9 帧、2 步 smoke：成功写出 \`outputs/smoke_open_source_20261012/20261012_0830_1/\`。
- 49 帧、4 步标准时序 smoke：成功写出 \`outputs/smoke_open_source_49f_20261012/20261012_0837_1/\`。
- 49 帧结果包含 \`gen.mp4\`、\`render.mp4\`、\`mask.mp4\`、\`input.mp4\`、\`viz.mp4\`；\`gen.mp4\` 可读，49 帧、10 FPS、672×384。
- 日志确认已加载 expert-aware transformer、latent residual router、BLIP2、DepthCrafter，并完成扩散去噪和解码。
- LIBERO transformer 单独加载成功：\`CrossTransformer3DModel\`，5,646,225,984 参数，未启用 latent router。

## 服务器上的主要权重

- \`weights/checkpoint/transformer/\`：约 11GB，latent expert 主 transformer。
- \`weights/latent_residual_weighted/final/latent_residual_mse.pt\`：约 122MB，latent residual router。
- \`weights/libero_checkpoint/transformer/\`：约 11GB，LIBERO transformer。
- 外部基础模型位于 \`/home/dataset-assist-0/pytu/checkpoints/\`：CogVideoX、DepthCrafter、SVD、BLIP2。


### 清理结果

已删除训练专用和未被推理入口引用的内容：src/videox_fun/、src/train.py、scripts/train.sh、data/train.csv、重复的 src/crosstransformer3d.py、未使用的 latent_delta_router.py、DepthCrafter 独立推理示例，以及 Pi3/SAM3 的演示和基准脚本。Pi3 与 SAM3 的实际推理后端文件保留。

清理后重新运行 9 帧 smoke 成功，输出目录为：
outputs/smoke_after_cleanup_20261012/20261012_0911_1/

gen.mp4 校验通过：9 帧、10 FPS、672×384。

### input1.mp4 49 帧测试

输入：input1.mp4（49 帧、10 FPS、672×384）。
输出：outputs/input1_49f_20261012/20261012_0922_input1/。
结果：gen.mp4、input.mp4、mask.mp4、render.mp4 均为 49 帧、10 FPS、672×384；viz.mp4 为 97 帧、20 FPS、1374×384。

### input1.mp4 gradual 49 帧测试

使用 scripts/infer_gradual.sh（MODE=gradual）完成 49 帧推理。输出：outputs/input1_49f_gradual_20261012/20261012_0926_input1/。gen.mp4、input.mp4、mask.mp4、render.mp4 均为 49 帧、10 FPS、672×384；viz.mp4 为 97 帧、20 FPS、1374×384。

### input1.mp4 Pi3 gradual 49 帧测试

使用 Pi3X 本地权重 /home/dataset-assist-0/pytu/Embody4D/weights/Pi3X/model.safetensors，warp_backend=pi3，MODE=gradual，完成 49 帧推理。输出：outputs/input1_49f_pi3_gradual_20261012/20261012_0945_input1/。gen.mp4、input.mp4、mask.mp4、render.mp4 均为 49 帧、10 FPS、672×384；viz.mp4 为 97 帧、20 FPS、1374×384。Pi3 中间 warp 文件 warped_rgb.mp4 / warped_mask.mp4 也均为 49 帧。

Pi3 测试使用了带 FFN expert router 的项目 transformer，因此 ENABLE_LATENT_EXPERTS=1；脚本默认值仍为 0。
