#!/usr/bin/env bash
# Ré-exécute tous les notebooks dans l'ordre (régénère figures/ et results/).
# Durée indicative : ~25-30 min (la calibration Heston et le notebook 07 sont les plus longs).
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
for nb in 0*.ipynb; do
  echo ">>> $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done
