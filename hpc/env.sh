# Shared environment for every BluePebble job. Sourced, not executed.
# Fill in the two placeholders below once (see hpc/README.md for how to discover them).

export ACRC_ACCOUNT="CHANGE_ME_project_account"
export ACRC_GPU_PARTITION="gpu"
export ACRC_CPU_PARTITION="compute"

REPO="/user/work/${USER}/BristolProject"

# Caches live on /user/work: home quotas are too small for 15 GB of crops and HF weights.
export HF_HOME="/user/work/${USER}/hf_cache"
export TORCH_HOME="/user/work/${USER}/torch_cache"
export TIMM_HOME="${TORCH_HOME}"
export PYTHONUNBUFFERED=1

# One thread pool per GPU job; oversubscription on shared nodes slows everyone down.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
# Adjust to whatever `module avail` offers; a recent CUDA-enabled Python is all that is needed.
module load languages/python/3.12.3 2>/dev/null || module load languages/anaconda3 2>/dev/null || true

if [ -d "${REPO}/.venv" ]; then
    source "${REPO}/.venv/bin/activate"
fi

cd "${REPO}" || exit 1   # every data path in the code is relative to the repository root
