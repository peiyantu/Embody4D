#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PI3_REPO_URL="${PI3_REPO_URL:-https://github.com/yyfz/Pi3.git}"
PI3_COMMIT="${PI3_COMMIT:-b56ef4b9cc9be05b8590604861aa3906db0ef45e}"
TARGET_DIR="${PI3_TARGET_DIR:-${ROOT_DIR}/third_party/Pi3}"
BRIDGE_SRC="${ROOT_DIR}/scripts/pi3_bridge/run_pi3_warp_two_videos_firstframe_pose.py"
BRIDGE_DST="${TARGET_DIR}/run_pi3_warp_two_videos_firstframe_pose.py"

if [[ -e "${TARGET_DIR}" && ! -d "${TARGET_DIR}/.git" ]]; then
  if [[ -f "${BRIDGE_DST}" ]]; then
    echo "Pi3 already exists at ${TARGET_DIR}"
    echo "Using vendored working tree; not overwriting it."
    exit 0
  fi
  echo "Target exists but is not a Pi3 checkout and lacks ${BRIDGE_DST}" >&2
  echo "Move it away or set PI3_TARGET_DIR to another path." >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone --recursive "${PI3_REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --tags origin "${PI3_COMMIT}" || git -C "${TARGET_DIR}" fetch --tags origin
git -C "${TARGET_DIR}" checkout "${PI3_COMMIT}"
git -C "${TARGET_DIR}" submodule update --init --recursive

if [[ ! -f "${BRIDGE_SRC}" ]]; then
  echo "Missing Embody4D Pi3 bridge script: ${BRIDGE_SRC}" >&2
  exit 1
fi
cp "${BRIDGE_SRC}" "${BRIDGE_DST}"

echo "Pi3 is ready at ${TARGET_DIR}"
echo "Pinned commit: ${PI3_COMMIT}"
