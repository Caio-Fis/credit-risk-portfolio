#!/usr/bin/env bash
# Sync this repo's API service into the matching HuggingFace Space.
#
# Prerequisite: the Space must already exist (create once via
# https://huggingface.co/new-space — SDK=Docker, port 7860).
#
# Auth: prefer HF_TOKEN env var so `make` doesn't have to forward stdin.
# Get the token at https://huggingface.co/settings/tokens (Write scope).
#
# Usage:
#   HF_TOKEN=hf_xxx make sync-hf-space
#   HF_TOKEN=hf_xxx HF_USER=Caio-Fis HF_SPACE=credit-risk-api bash scripts/sync_hf_space.sh

set -euo pipefail

HF_USER="${HF_USER:-Caio-Fis}"
HF_SPACE="${HF_SPACE:-credit-risk-api}"
SPACE_DIR="${SPACE_DIR:-$(mktemp -d -t credit-risk-hf-XXXXXX)}"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var not set." >&2
  echo "Run:   HF_TOKEN=hf_... make sync-hf-space" >&2
  echo "Token: https://huggingface.co/settings/tokens (Write scope)" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ERROR: git-lfs is required by HuggingFace Spaces for binary files." >&2
  echo "Install with:" >&2
  echo "  sudo dnf install git-lfs       # Fedora / RHEL" >&2
  echo "  sudo apt install git-lfs       # Debian / Ubuntu" >&2
  echo "  brew install git-lfs           # macOS" >&2
  exit 1
fi

echo "==> Cloning ${SPACE_URL} into ${SPACE_DIR}"
# Use the token in the clone URL too so the credential helper doesn't prompt.
AUTH_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
git clone "${AUTH_URL}" "${SPACE_DIR}"

echo "==> Enabling git-lfs and tracking model/parquet binaries"
git -C "${SPACE_DIR}" lfs install --local --skip-smudge >/dev/null
# Patterns that HF Spaces requires to go through LFS/Xet
git -C "${SPACE_DIR}" lfs track "*.joblib" "*.parquet" >/dev/null

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Copying Docker / service files"
cp "${REPO_ROOT}/Dockerfile" "${SPACE_DIR}/"
cp "${REPO_ROOT}/.dockerignore" "${SPACE_DIR}/"
cp "${REPO_ROOT}/pyproject.toml" "${SPACE_DIR}/"

echo "==> Copying src/ (excluding __pycache__)"
mkdir -p "${SPACE_DIR}/src"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "${REPO_ROOT}/src/" "${SPACE_DIR}/src/"
else
  cp -r "${REPO_ROOT}/src/." "${SPACE_DIR}/src/"
  find "${SPACE_DIR}/src" -type d -name __pycache__ -exec rm -rf {} +
  find "${SPACE_DIR}/src" -type f -name '*.pyc' -delete
fi

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

# Ensure pyc bytecode doesn't sneak back in on subsequent syncs.
cat > "${SPACE_DIR}/.gitignore" <<'GITIGNORE'
__pycache__/
*.pyc
*.pyo
.venv/
.env
GITIGNORE

cd "${SPACE_DIR}"
git add -A
SOURCE_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)
if git diff --cached --quiet; then
  echo "==> No changes to push (Space already in sync with ${SOURCE_SHA})."
else
  git -c user.name="credit-risk-portfolio sync" \
      -c user.email="sync@local" \
      commit -m "sync: ${SOURCE_SHA} from credit-risk-portfolio"
  echo "==> Pushing to Space"
  REMOTE_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo main)
  git push origin "${REMOTE_BRANCH}"
  echo
  echo "==> Done. Space URL: ${SPACE_URL}"
  echo "    Build status:   ${SPACE_URL}/logs"
  echo "    Live endpoint:  https://${HF_USER}-${HF_SPACE}.hf.space"
fi
