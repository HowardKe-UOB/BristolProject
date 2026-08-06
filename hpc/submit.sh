#!/bin/bash
# Submit a job with the account and partition from hpc/env.sh, so the .sbatch files
# themselves stay free of site-specific values.
#
#   bash hpc/submit.sh hpc/03_cap.sbatch
#   bash hpc/submit.sh hpc/04_teacher.sbatch --dependency=afterok:1234567
set -euo pipefail

script="${1:?usage: bash hpc/submit.sh <script.sbatch> [extra sbatch args]}"
shift || true

source "$(dirname "$0")/env.sh"

if [ "${ACRC_ACCOUNT}" = "CHANGE_ME_project_account" ]; then
    echo "Set ACRC_ACCOUNT in hpc/env.sh first. Discover it with:" >&2
    echo "    sacctmgr -n show assoc user=\$USER format=account%30" >&2
    exit 1
fi

# GPU jobs are the ones that ask for a gres; everything else goes to the CPU partition.
if grep -q "gres=gpu" "${script}"; then
    partition="${ACRC_GPU_PARTITION}"
else
    partition="${ACRC_CPU_PARTITION}"
fi

echo "sbatch --account=${ACRC_ACCOUNT} --partition=${partition} $* ${script}"
sbatch --account="${ACRC_ACCOUNT}" --partition="${partition}" "$@" "${script}"
