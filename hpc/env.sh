# Shared environment for every BluePebble job. Sourced, not executed.
# Fill in the two placeholders below once (see hpc/README.md for how to discover them).

# The ":-" form means a value already in the environment wins, so a one-off run can override
# any of these without editing the file:
#     ACRC_GPU_GRES="gpu:a100:1" bash hpc/submit.sh hpc/03_cap.sbatch
export ACRC_ACCOUNT="${ACRC_ACCOUNT:-CHANGE_ME_project_account}"
export ACRC_GPU_PARTITION="${ACRC_GPU_PARTITION:-gpu}"
export ACRC_CPU_PARTITION="${ACRC_CPU_PARTITION:-compute}"

# Which GPU to ask for. The default batch (--P 12 --K 4 --T 2, i.e. 96 crops of 518x518
# through ViT-B) needs roughly 20 GB, so an 11 GB RTX 2080 Ti runs out of memory, and a bare
# "gpu:1" lets the scheduler hand you one of those. Name the type instead. Exact strings:
#     sinfo -N -p "${ACRC_GPU_PARTITION}" -o "%N %G" | sort -u -k2
export ACRC_GPU_GRES="${ACRC_GPU_GRES:-gpu:1}"

# Derived from this file's own location, so the repository can live anywhere.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Caches sit beside the code, which on BluePebble means /user/work: home quotas are far too
# small for 15 GB of crops and several GB of pretrained weights. Both are in .gitignore.
export HF_HOME="${REPO}/hf_cache"
export TORCH_HOME="${REPO}/torch_cache"
export TIMM_HOME="${TORCH_HOME}"
export PYTHONUNBUFFERED=1

# One thread pool per GPU job; oversubscription on shared nodes slows everyone down.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Guarded so these scripts also run on a machine with no module system (a laptop, say).
if command -v module >/dev/null 2>&1; then
    module purge
    # Adjust to whatever `module avail` offers; a recent CUDA-capable Python is all it needs.
    module load languages/python/3.12.3 2>/dev/null \
        || module load languages/anaconda3 2>/dev/null || true
fi

if [ -d "${REPO}/.venv" ]; then
    source "${REPO}/.venv/bin/activate"
fi

cd "${REPO}" || exit 1   # every data path in the code is relative to the repository root
