#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

die() { echo "$*" >&2; exit 1; }
need_dir() { [[ -d "$1" ]] || die "$2 not found: $1"; }
need_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
VIDEO_PATH="${VIDEO_PATH:-}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/libero_infer}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/weights/libero_checkpoint}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${CHECKPOINT_DIR}/transformer}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${ROOT_DIR}/checkpoints}"
MODEL_NAME="${MODEL_NAME:-${PRETRAINED_ROOT}/CogVideoX-Fun-V1.1-5b-InP}"
BLIP_PATH="${BLIP_PATH:-${ROOT_DIR}/weights/blip2-opt-2.7b}"
UNET_PATH="${UNET_PATH:-${PRETRAINED_ROOT}/DepthCrafter}"
PRE_TRAIN_PATH="${PRE_TRAIN_PATH:-${PRETRAINED_ROOT}/stable-video-diffusion-img2vid}"
TRAJ_TXT="${TRAJ_TXT:-${ROOT_DIR}/assets/trajs/loop2.txt}"
PI3_REPO_PATH="${PI3_REPO_PATH:-${ROOT_DIR}/third_party/Pi3}"
SAMPLE_SIZE_H="${SAMPLE_SIZE_H:-256}"
SAMPLE_SIZE_W="${SAMPLE_SIZE_W:-256}"
MODE="${MODE:-gradual}"
DEVICE="${DEVICE:-cuda:0}"
CAMERA="${CAMERA:-target}"

case "${MODE}" in
  gradual|direct|bullet|zoom) ;;
  *) die "Unsupported MODE=${MODE}. Expected gradual, direct, bullet, or zoom." ;;
esac

[[ -n "${VIDEO_PATH}" ]] || die "Set VIDEO_PATH=/path/to/video.mp4 or pass the video as the first argument."
need_file "${VIDEO_PATH}" "Input video"
need_dir "${TRANSFORMER_PATH}" "LIBERO transformer path"
need_dir "${MODEL_NAME}" "Base model path"
need_dir "${BLIP_PATH}" "BLIP path"
need_dir "${PI3_REPO_PATH}" "Pi3 repo path"

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
  --device "${DEVICE}"
  --warp_backend pi3
  --pi3_repo_path "${PI3_REPO_PATH}"
  --sample_size "${SAMPLE_SIZE_H}" "${SAMPLE_SIZE_W}"
  --max_res "${MAX_RES:-256}"
)

[[ "${PI3_LOCAL_FILES_ONLY:-1}" == "1" ]] && cmd+=(--pi3_local_files_only)
add_if_set --exp_name "${EXP_NAME:-}"
add_if_set --prompt "${PROMPT:-}"
add_if_set --refvideo "${REFVIDEO:-}"
add_if_set --warp_path "${WARP_PATH:-}"
add_if_set --mask_path "${MASK_PATH:-}"

cmd+=("$@")
run_cmd
