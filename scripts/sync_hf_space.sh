#!/usr/bin/env bash
# Sync this repo's API service into the matching HuggingFace Space.
#
# Prerequisite: the Space must already exist (create once via
# https://huggingface.co/new-space — SDK=Docker, port 7860).
#
# Auth: this script delegates to your git credentials. If the push asks
# for a password, paste your HF access token (huggingface.co/settings/tokens
# — needs "Write" scope on Spaces).
#
# Usage:
#   bash scripts/sync_hf_space.sh
#   HF_USER=Caio-Fis HF_SPACE=credit-risk-api bash scripts/sync_hf_space.sh

set -euo pipefail

HF_USER="${HF_USER:-Caio-Fis}"
HF_SPACE="${HF_SPACE:-credit-risk-api}"
SPACE_DIR="${SPACE_DIR:-$(mktemp -d -t credit-risk-hf-XXXXXX)}"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"

echo "==> Cloning ${SPACE_URL} into ${SPACE_DIR}"
git clone "${SPACE_URL}" "${SPACE_DIR}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Copying Docker / service files"
cp "${REPO_ROOT}/Dockerfile" "${SPACE_DIR}/"
cp "${REPO_ROOT}/.dockerignore" "${SPACE_DIR}/"
cp "${REPO_ROOT}/pyproject.toml" "${SPACE_DIR}/"

mkdir -p "${SPACE_DIR}/src"
cp -r "${REPO_ROOT}/src/." "${SPACE_DIR}/src/"

echo "==> Copying model artefacts (joblib + macro + monitor sources)"
mkdir -p "${SPACE_DIR}/artifacts" "${SPACE_DIR}/data/processed" "${SPACE_DIR}/data/schemas"
cp "${REPO_ROOT}/artifacts/pd_model_lc.joblib" "${SPACE_DIR}/artifacts/"
cp "${REPO_ROOT}/artifacts/pd_calibrator_lc.joblib" "${SPACE_DIR}/artifacts/"
cp "${REPO_ROOT}/data/processed/macro_features.parquet" "${SPACE_DIR}/data/processed/"
cp "${REPO_ROOT}/data/processed/arf_drifts_lc.csv" "${SPACE_DIR}/data/processed/"
cp "${REPO_ROOT}/data/processed/sliding_calibration_lc.csv" "${SPACE_DIR}/data/processed/"
cp "${REPO_ROOT}/data/schemas/lendingclub.json" "${SPACE_DIR}/data/schemas/"

echo "==> Installing HuggingFace-flavoured README at Space root"
cp "${REPO_ROOT}/hf_space/README.md" "${SPACE_DIR}/README.md"

cd "${SPACE_DIR}"
git add -A
SOURCE_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)
if git diff --cached --quiet; then
  echo "==> No changes to push (Space already in sync with ${SOURCE_SHA})."
else
  git commit -m "sync: ${SOURCE_SHA} from credit-risk-portfolio"
  echo "==> Pushing to Space (paste your HF write token if prompted)"
  # HF Spaces default branch is `main` for new spaces.
  REMOTE_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo main)
  git push origin "${REMOTE_BRANCH}"
  echo
  echo "==> Done. Space URL: ${SPACE_URL}"
  echo "    Build status:   ${SPACE_URL}/logs"
  echo "    Live endpoint:  https://${HF_USER}-${HF_SPACE}.hf.space"
fi
