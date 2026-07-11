#!/usr/bin/env bash
#
# download_data.sh — convenience wrapper around src.data.ingestion.
#
# Downloads the raw UCI Heart Disease dataset to the path configured in
# src/config.py (RAW_DATA_PATH), then verifies the expected columns are
# present. Honors DATA_URL / RAW_DATA_PATH environment overrides.
#
# Usage:
#   ./scripts/download_data.sh
#
set -euo pipefail

# Resolve repo root as the parent of this script's directory so the script
# works regardless of the current working directory (no hardcoded paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python -m src.data.ingestion --verify "$@"
