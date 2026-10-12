#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

die() { echo "$*" >&2; exit 1; }
need_dir() { [[ -d "$1" ]] || die "$2 not found: $1"; }
need_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
need_exec() { [[ -x "$1" ]] || command -v "$1" >/dev/null 2>&1 || die "$2 not found or not executable: $1"; }
resolve_project_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  elif [[ -e "${ROOT_DIR}/$value" ]]; then
    local value_no_dot
    value_no_dot="$value"
    value_no_dot="${value_no_dot#./}"
    printf '%s/%s' "$ROOT_DIR" "$value_no_dot"
  else
    printf '%s' "$value"
  fi
}
add_if_set() { [[ -n "${2:-}" ]] && cmd+=("$1" "$2"); return 0; }
run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    exec "${cmd[@]}"
  fi
}

if [[ -z "${VIDEO_PATH:-}" && $# -gt 0 && "${1:0:1}" != "-" ]]; then
  VIDEO_PATH="$1"
  shift
fi

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  . "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME:-base}"
  set -u
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3 python3.10 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi
[[ -n "${PYTHON_BIN:-}" ]] || die "Python interpreter not found; set PYTHON_BIN=/path/to/python."
VIDEO_PATH="${VIDEO_PATH:-}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/infer_ckpt5000}"
VIDEO_PATH="$(resolve_project_path "$VIDEO_PATH")"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${ROOT_DIR}/${OUT_DIR#./}"
fi
ENABLE_LATENT_EXPERTS="${ENABLE_LATENT_EXPERTS:-0}"
case "${ENABLE_LATENT_EXPERTS}" in
  0|1) ;;
  *) die "Unsupported ENABLE_LATENT_EXPERTS=${ENABLE_LATENT_EXPERTS}. Expected 0 or 1." ;;
esac
DEFAULT_CHECKPOINT_DIR="${ROOT_DIR}/weights/checkpoint"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
if [[ -e "${CHECKPOINT_DIR}/transformer" ]]; then
  DEFAULT_TRANSFORMER_PATH="${CHECKPOINT_DIR}/transformer"
else
  DEFAULT_TRANSFORMER_PATH="${CHECKPOINT_DIR}"
fi
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${DEFAULT_TRANSFORMER_PATH}}"
LATENT_RESIDUAL_CHECKPOINT="${LATENT_RESIDUAL_CHECKPOINT:-${ROOT_DIR}/weights/latent_residual_weighted/final}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${ROOT_DIR}/checkpoints}"
PRETRAINED_ROOT="$(resolve_project_path "$PRETRAINED_ROOT")"
MODEL_NAME="${MODEL_NAME:-${PRETRAINED_ROOT}/CogVideoX-Fun-V1.1-5b-InP}"
BLIP_PATH="${BLIP_PATH:-${ROOT_DIR}/weights/blip2-opt-2.7b}"
UNET_PATH="${UNET_PATH:-${PRETRAINED_ROOT}/DepthCrafter}"
PRE_TRAIN_PATH="${PRE_TRAIN_PATH:-${PRETRAINED_ROOT}/stable-video-diffusion-img2vid}"
TRAJ_TXT="${TRAJ_TXT:-${ROOT_DIR}/assets/trajs/loop2.txt}"
CHECKPOINT_DIR="$(resolve_project_path "$CHECKPOINT_DIR")"
TRANSFORMER_PATH="$(resolve_project_path "$TRANSFORMER_PATH")"
LATENT_RESIDUAL_CHECKPOINT="$(resolve_project_path "$LATENT_RESIDUAL_CHECKPOINT")"
MODEL_NAME="$(resolve_project_path "$MODEL_NAME")"
BLIP_PATH="$(resolve_project_path "$BLIP_PATH")"
UNET_PATH="$(resolve_project_path "$UNET_PATH")"
PRE_TRAIN_PATH="$(resolve_project_path "$PRE_TRAIN_PATH")"
TRAJ_TXT="$(resolve_project_path "$TRAJ_TXT")"
MODE="${MODE:-gradual}"
DEVICE="${DEVICE:-cuda:0}"
CAMERA="${CAMERA:-target}"
WARP_BACKEND="${WARP_BACKEND:-pi3}"
PI3_REPO_PATH="${PI3_REPO_PATH:-${ROOT_DIR}/third_party/Pi3}"
PI3_CKPT="${PI3_CKPT:-}"
PI3_CACHE_DIR="${PI3_CACHE_DIR:-}"
PI3_LOCAL_FILES_ONLY="${PI3_LOCAL_FILES_ONLY:-0}"

# Front guidance is opt-in so latent-route evals do not accidentally inherit it.
ENABLE_FRONT_GUIDANCE="${ENABLE_FRONT_GUIDANCE:-0}"
USE_SAM3_FRONT_MASK="${USE_SAM3_FRONT_MASK:-0}"
SAM3_REPO_PATH="${SAM3_REPO_PATH:-${ROOT_DIR}/third_party/sam3}"
if [[ -z "${SAM3_PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/../sam3/bin/python" ]]; then
  SAM3_PYTHON_BIN="${CONDA_PREFIX}/../sam3/bin/python"
else
  SAM3_PYTHON_BIN="${SAM3_PYTHON_BIN:-${PYTHON_BIN}}"
fi
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-${ROOT_DIR}/weights/sam3/sam3.pt}"
SAM3_REPO_PATH="$(resolve_project_path "$SAM3_REPO_PATH")"
SAM3_CHECKPOINT_PATH="$(resolve_project_path "$SAM3_CHECKPOINT_PATH")"
SAM3_PROMPTS="${SAM3_PROMPTS:-object|robot arm|moving object}"
SAM3_PROMPT_FRAME_INDEX="${SAM3_PROMPT_FRAME_INDEX:-0}"

case "${MODE}" in
  gradual|direct|bullet|zoom) ;;
  *) die "Unsupported MODE=${MODE}. Expected gradual, direct, bullet, or zoom." ;;
esac
case "${ENABLE_FRONT_GUIDANCE}" in
  0|1) ;;
  *) die "Unsupported ENABLE_FRONT_GUIDANCE=${ENABLE_FRONT_GUIDANCE}. Expected 0 or 1." ;;
esac
case "${USE_SAM3_FRONT_MASK}" in
  0|1) ;;
  *) die "Unsupported USE_SAM3_FRONT_MASK=${USE_SAM3_FRONT_MASK}. Expected 0 or 1." ;;
esac

[[ -n "${VIDEO_PATH}" ]] || die "Set VIDEO_PATH=/path/to/video.mp4 or pass the video as the first argument."
need_file "${VIDEO_PATH}" "Input video"
need_dir "${TRANSFORMER_PATH}" "Transformer path"
if [[ "${WARP_BACKEND}" == "pi3" ]]; then
  need_dir "${PI3_REPO_PATH}" "Pi3 repo path"
fi
if [[ "${ENABLE_LATENT_EXPERTS}" == "1" ]]; then
  need_file "${LATENT_RESIDUAL_CHECKPOINT}/latent_residual_mse.pt" "Latent residual checkpoint"
fi
if [[ "${ENABLE_FRONT_GUIDANCE}" == "1" && "${USE_SAM3_FRONT_MASK}" == "1" ]]; then
  need_dir "${SAM3_REPO_PATH}" "SAM3 repo path"
  need_exec "${SAM3_PYTHON_BIN}" "SAM3 python"
  need_file "${SAM3_CHECKPOINT_PATH}" "SAM3 checkpoint"
fi

cd "${SRC_DIR}"

cmd=(
  "${PYTHON_BIN}" inference.py
  --video_path "${VIDEO_PATH}"
  --out_dir "${OUT_DIR}"
  --stride "${STRIDE:-1}"
  --radius_scale "${RADIUS_SCALE:-1.0}"
  --camera "${CAMERA}"
  --target_pose "${TARGET_POSE:-0 -10 0 0 0}"
  --mode "${MODE}"
  --model_name "${MODEL_NAME}"
  --blip_path "${BLIP_PATH}"
  --unet_path "${UNET_PATH}"
  --pre_train_path "${PRE_TRAIN_PATH}"
  --transformer_path "${TRANSFORMER_PATH}"
  --traj_txt "${TRAJ_TXT}"
  --video_length "${VIDEO_LENGTH:-49}"
  --fps "${FPS:-10}"
  --seed "${SEED:-43}"
  --sample_size "${SAMPLE_HEIGHT:-384}" "${SAMPLE_WIDTH:-672}"
  --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-50}"
  --depth_inference_steps "${DEPTH_INFERENCE_STEPS:-5}"
  --device "${DEVICE}"
  --warp_backend "${WARP_BACKEND}"
)

if [[ "${WARP_BACKEND}" == "pi3" ]]; then
  cmd+=(--pi3_repo_path "${PI3_REPO_PATH}")
  add_if_set --pi3_ckpt "${PI3_CKPT}"
  add_if_set --pi3_cache_dir "${PI3_CACHE_DIR}"
  [[ "${PI3_LOCAL_FILES_ONLY}" == "1" ]] && cmd+=(--pi3_local_files_only)
fi


if [[ "${ENABLE_LATENT_EXPERTS}" == "1" ]]; then
  cmd+=(
    --latent_residual_checkpoint="${LATENT_RESIDUAL_CHECKPOINT}"
    --latent_route_top_k="${LATENT_ROUTE_TOP_K:-2}"
    --latent_route_secondary_weight="${LATENT_ROUTE_SECONDARY_WEIGHT:-0.20}"
    --latent_route_copy_valid_threshold="${LATENT_ROUTE_COPY_VALID_THRESHOLD:-0.65}"
    --latent_route_copy_delta_threshold="${LATENT_ROUTE_COPY_DELTA_THRESHOLD:-0.50}"
    --latent_route_inpaint_delta_threshold="${LATENT_ROUTE_INPAINT_DELTA_THRESHOLD:-0.60}"
    --latent_route_inpaint_valid_threshold="${LATENT_ROUTE_INPAINT_VALID_THRESHOLD:-0.45}"
    --latent_route_force_inpaint_delta_threshold="${LATENT_ROUTE_FORCE_INPAINT_DELTA_THRESHOLD:-0.85}"
    --latent_route_postprocess_mode="${LATENT_ROUTE_POSTPROCESS_MODE:-legacy}"
    --latent_route_smooth_spatial_kernel="${LATENT_ROUTE_SMOOTH_SPATIAL_KERNEL:-1}"
    --latent_route_smooth_temporal_kernel="${LATENT_ROUTE_SMOOTH_TEMPORAL_KERNEL:-1}"
    --latent_route_valid_smooth_spatial_kernel="${LATENT_ROUTE_VALID_SMOOTH_SPATIAL_KERNEL:-1}"
    --latent_route_vote_spatial_kernel="${LATENT_ROUTE_VOTE_SPATIAL_KERNEL:-1}"
    --latent_route_vote_temporal_kernel="${LATENT_ROUTE_VOTE_TEMPORAL_KERNEL:-1}"
    --latent_ffn_expert_scale="${LATENT_FFN_EXPERT_SCALE:-1.0}"
    --latent_route_copy_residual_scale="${LATENT_ROUTE_COPY_RESIDUAL_SCALE:-1.0}"
  )
fi

if [[ "${ENABLE_FRONT_GUIDANCE}" == "1" ]]; then
  cmd+=(
    --enable_front_guidance
    --front_guidance_mode=foreground_memory
    --front_guidance_target_scale=0.0
    --front_guidance_boost_strength="${FRONT_GUIDANCE_BOOST_STRENGTH:-2.0}"
    --front_guidance_min_area="${FRONT_GUIDANCE_MIN_AREA:-0.002}"
    --front_guidance_max_area="${FRONT_GUIDANCE_MAX_AREA:-0.60}"
    --front_guidance_gate_floor="${FRONT_GUIDANCE_GATE_FLOOR:-0.0}"
    --front_memory_high_threshold="${FRONT_MEMORY_HIGH_THRESHOLD:-0.55}"
    --front_memory_quality_threshold="${FRONT_MEMORY_QUALITY_THRESHOLD:-0.01}"
    --front_memory_noise_threshold="${FRONT_MEMORY_NOISE_THRESHOLD:-0.08}"
    --front_memory_min_alignment="${FRONT_MEMORY_MIN_ALIGNMENT:-0.0}"
    --front_memory_quality_power="${FRONT_MEMORY_QUALITY_POWER:-1.0}"
    --front_memory_residual_ratio_max="${FRONT_MEMORY_RESIDUAL_RATIO_MAX:--1}"
    --front_memory_query_mask_power="${FRONT_MEMORY_QUERY_MASK_POWER:-0}"
    --front_memory_query_mask_floor="${FRONT_MEMORY_QUERY_MASK_FLOOR:-0}"
    --front_memory_train_gate_gradient_multiplier="${FRONT_MEMORY_TRAIN_GATE_GRADIENT_MULTIPLIER:-1.0}"
    --front_memory_max_area_std="${FRONT_MEMORY_MAX_AREA_STD:-0.1435}"
  )
fi

if [[ "${ENABLE_LATENT_EXPERTS}" == "1" && "${LATENT_ROUTE_POSTPROCESS_ON_VIDEO_GRID:-0}" == "1" ]]; then
  cmd+=(--latent_route_postprocess_on_video_grid)
fi

add_if_set --exp_name "${EXP_NAME:-}"
add_if_set --prompt "${PROMPT:-}"
add_if_set --refvideo "${REFVIDEO:-}"
add_if_set --warp_path "${WARP_PATH:-}"
add_if_set --mask_path "${MASK_PATH:-}"

if [[ "${ENABLE_FRONT_GUIDANCE}" == "1" ]]; then
  if [[ -n "${FRONTVIDEO:-}" ]]; then
    cmd+=(--frontvideo "${FRONTVIDEO}")
  elif [[ "${USE_SAM3_FRONT_MASK}" == "1" ]]; then
    IFS='|' read -r -a SAM3_PROMPT_ARRAY <<< "${SAM3_PROMPTS}"
    cmd+=(
      --sam3_front_mask
      --sam3_repo_path "${SAM3_REPO_PATH}"
      --sam3_python_bin "${SAM3_PYTHON_BIN}"
      --sam3_checkpoint_path "${SAM3_CHECKPOINT_PATH}"
      --sam3_prompt_frame_index "${SAM3_PROMPT_FRAME_INDEX}"
      --sam3_reuse_mask
      --sam3_prompts "${SAM3_PROMPT_ARRAY[@]}"
    )
  else
    die "Set FRONTVIDEO or USE_SAM3_FRONT_MASK=1, or set ENABLE_FRONT_GUIDANCE=0."
  fi
fi

cmd+=("$@")
run_cmd
