#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_subject_pipeline.sh hyq_0402
#   bash run_subject_pipeline.sh chaoran_0529
#   bash run_subject_pipeline.sh zhenhong_0529 --no-stag --skip-visualize
#
# You can also edit DEFAULT_SUBJECT below and run:
#   bash run_subject_pipeline.sh

DEFAULT_SUBJECT="lisha_0407"
CONDA_ENV="baroposer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUBJECT="${1:-$DEFAULT_SUBJECT}"
if [[ -z "$SUBJECT" ]]; then
  echo "Usage: bash run_subject_pipeline.sh <subject> [pipeline args...]"
  echo "Example: bash run_subject_pipeline.sh hyq_0402"
  echo "Example: bash run_subject_pipeline.sh chaoran_0529"
  exit 2
fi

shift || true

conda run -n "$CONDA_ENV" python run_subject_pipeline.py \
  --subject "$SUBJECT" \
  "$@"
