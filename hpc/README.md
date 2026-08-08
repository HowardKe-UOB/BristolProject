# Running the pipeline on BluePebble (Bristol ACRC)

These scripts run the full reproduction ladder as Slurm jobs. They assume the layout of
BluePebble: `bp1-login*` login nodes, `/user/home/$USER` (small quota) and
`/user/work/$USER` (large, where everything below lives).

Placeholders to fill in once, at the top of `env.sh`: your **project account** and the
**GPU partition name**. Discover both with:

```bash
sinfo -o "%20P %10G %30f %N"          # partitions, gres, features, nodes
sacctmgr -n show assoc user=$USER format=account%30   # accounts you may charge to
```

## Before you start: three things that are easy to get wrong

**1. Pick a 24 GB GPU.** BluePebble's GPU nodes mix RTX 2080 Ti (11 GB), RTX 3090 (24 GB)
and V100. The reported runs used a 24 GB card with the default batch (`--P 12 --K 4 --T 2`,
i.e. 96 crops of 518x518 through ViT-B). On an 11 GB 2080 Ti that will run out of memory.
Either constrain the job to a 3090/V100-32GB, or halve `--P` and say so, because a changed
batch size is a changed training recipe.

**2. Compute nodes may have no outbound internet.** The backbones (DINOv2 via timm,
MegaDescriptor via Hugging Face) are downloaded on first use. Run `setup_env.sh` on the
**login node** once: it installs the environment and pre-downloads every weight into
`/user/work/$USER/hf_cache`. The job scripts then set `HF_HUB_OFFLINE=1`.

**3. Budget about 100 GB of disk on `/user/work`.** The image cache is 15 GB; the core
ladder's checkpoints are roughly 20 GB and the extension rungs (thirteen Mega-sized
checkpoints at ~1.3 GB each) add another ~17 GB; extracted crops, embeddings and the
HF cache take the rest.

## Which cards actually work here

Measured, not assumed. The smoke test was run against each type.

| Card | Memory | Verdict |
|---|---|---|
| A100 (`gpu:a100:1`) | 20.9 GB as a `1g.20gb` MIG slice | works; 18.8 GB peak leaves 2.2 GB spare |
| RTX 3090 (`gpu:rtx_3090:1`) | 24 GB | fits, one node of 8 cards |
| RTX 2080 Ti (`gpu:rtx_2080:1`) | 11 GB | too small to train; fine for step 02, which peaks at 0.8 GB |
| V100 (`gpu:v100:1` / `gpu:V100:1`) | **32 GB, and still unusable** | see below |

The V100s have ample memory but are compute capability 7.0, and the torch this cluster
resolves (2.13.0+cu130) ships kernels only for 7.5 and above. Every GPU call fails with
`no kernel image is available for execution on the device`. To use them, reinstall torch
against an older CUDA build inside `.venv`, which also keeps A100 and RTX support:

```
pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

That is worth doing only if the A100 queue is long: it unlocks nine 32 GB cards and removes
the tight 2.2 GB headroom, at the cost of re-resolving the environment mid-project.

Note also that this site spells the same card two ways, `gpu:V100:3` on bp1-gpu036 and
bp1-gpu040 but `gpu:v100:3` on bp1-gpu037. Gres names are case-sensitive, so a request for
one spelling can never be scheduled on the other nodes.

## One-time setup (login node)

```bash
cd /user/work/$USER
git clone https://github.com/HowardKe-UOB/BristolProject.git
cd BristolProject

# put the dataset here (about 900 MB; scp from your laptop or copy from RDSF)
#   2025Sep18.tar.gz
#   2025Sep18.listing.txt

bash hpc/setup_env.sh          # venv + dependencies + pre-downloaded backbones
```

## Submitting the ladder

Each script prints the `sbatch` line for the next one. Run them from the repository root.

```bash
bash hpc/submit.sh hpc/01_signals_cache.sbatch    # CPU  ~5 m   signals, splits, 15 GB image cache
bash hpc/submit.sh hpc/02_vits_cache.sbatch       # GPU  ~6 m   frozen ViT-S cache Stage 1 reads
bash hpc/submit.sh hpc/03_cap.sbatch              # GPU  array 0-4, Stage 1 (five seeds)
bash hpc/submit.sh hpc/04_teacher.sbatch          # GPU  ~30 m  ensemble teacher + frozen labels
bash hpc/submit.sh hpc/05a_holdout_students.sbatch # GPU array 0-4, holdout students (seeds 5-9)
bash hpc/submit.sh hpc/05b_student_space.sbatch   # GPU  ~20 m  embeds them into the v4 mining space
bash hpc/submit.sh hpc/05_students.sbatch         # GPU  array 0-5, deployment + hard-CL students
bash hpc/submit.sh hpc/06_mega.sbatch             # GPU  array 0-2, three Mega students
bash hpc/submit.sh hpc/06b_embed_sweep.sbatch     # GPU  ~50 m  embeds the nine students for fusion
bash hpc/submit.sh hpc/07_fuse_eval.sbatch        # CPU  ~35 m  fusion and the three protocols
```

That reproduces the core ladder (0.926-tier). The 0.945 selection needs the extension
rungs, chained the same way: `08a` fused teacher -> `08b` seven Mega variants ->
`08c` embed + rung-2 super teacher -> `08d` sup2 students -> `08e` embed + rung-3
teacher -> `08f` rung-3 students -> `08g` embed + every selection script (prints the
headline `max_P1`; reference 0.9448).

Steps 3 to 6 depend on the previous step finishing. Chain them if you prefer not to watch:

```bash
J=$(sbatch --parsable hpc/03_cap.sbatch)
J=$(sbatch --parsable --dependency=afterok:$J hpc/04_teacher.sbatch)
J=$(sbatch --parsable --dependency=afterok:$J hpc/05_students.sbatch)
J=$(sbatch --parsable --dependency=afterok:$J hpc/06_mega.sbatch)
sbatch --dependency=afterok:$J hpc/07_fuse_eval.sbatch
```

## Walltime is not a problem

The trainers checkpoint to disk every chunk and resume from where they stopped: each call
trains for `--wall` seconds, saves, and exits. The job scripts loop on this until the target
step count is reached, and a job that hits its time limit can simply be resubmitted, picking
up from the saved step. Nothing is lost and no run has to fit inside one allocation.

## Checking the results

`07_fuse_eval.sbatch` writes the same JSON files that are archived in `artifacts2/`. Compare
your run against the committed ones:

```bash
python - <<'PY'
import json
mine = json.load(open("artifacts2/fuse_final_v1.json"))
print(json.dumps(mine, indent=1)[:800])
PY
```

Expect agreement within the reported seed variance (students +-0.03 rank-1), not bit-exact
equality: a different GPU, a different cuDNN version and a different data-loading order all
perturb the fifth decimal place. The claim being reproduced is the ladder's shape, from 0.50
at the bottom to 0.94 at the top.
