#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
SAM3_COMMIT="${SAM3_COMMIT:-11dec2936de97f2857c1f76b66d982d5a001155d}"
TARGET_DIR="${SAM3_TARGET_DIR:-${ROOT_DIR}/third_party/sam3}"
WEIGHT_PATH="${SAM3_CHECKPOINT_PATH:-${ROOT_DIR}/weights/sam3/sam3.pt}"

if [[ -e "${TARGET_DIR}" && ! -d "${TARGET_DIR}/.git" ]]; then
  echo "SAM3 already exists at ${TARGET_DIR}"
  echo "Using vendored working tree; not overwriting it."
else
  if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${TARGET_DIR}")"
    git clone --recursive "${SAM3_REPO_URL}" "${TARGET_DIR}"
  fi
  git -C "${TARGET_DIR}" fetch --tags origin "${SAM3_COMMIT}" || git -C "${TARGET_DIR}" fetch --tags origin
  git -C "${TARGET_DIR}" checkout "${SAM3_COMMIT}"
  git -C "${TARGET_DIR}" submodule update --init --recursive
fi

mkdir -p "$(dirname "${WEIGHT_PATH}")"
if [[ -f "${WEIGHT_PATH}" ]]; then
  echo "SAM3 checkpoint found at ${WEIGHT_PATH}"
else
  echo "SAM3 checkpoint is not present yet: ${WEIGHT_PATH}" >&2
  echo "Place sam3.pt there or set SAM3_CHECKPOINT_PATH to your local checkpoint." >&2
fi
