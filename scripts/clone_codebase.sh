#!/bin/bash
# Usage: source scripts/clone_codebase.sh
# Purpose: create a fresh copy of the codebase from which we can safely launch jobs

SLURM_SUBMIT=${1-0}
TIMESTAMP=$(date +"%y%m%d%H%M%S")
DEST_DIR="/home/$USER/launch/bagel_${TIMESTAMP}"

### Create a copy of the code and switch directory
mkdir -p "${DEST_DIR}"

rsync -zarv \
--exclude="*.ipynb" \
--exclude="*.gif" \
--exclude="*.jpg" \
--exclude="*.png" \
--exclude="*.so" \
--exclude="__pycache__" \
--exclude="build" \
--exclude='*.egg' \
--exclude='*.log' \
--exclude='.git' \
. "${DEST_DIR}"

chmod -R 700 "${DEST_DIR}"

cd "${DEST_DIR}"

if [ "${SLURM_SUBMIT}" -eq 1 ]; then
    sbatch scripts/multinodes_run.sh
fi
