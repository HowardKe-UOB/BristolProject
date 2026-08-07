# Pairwise, Not Transitive: Label-Free Multi-Camera Cattle Re-Identification

Code for a fully **label-free** cross-camera cattle re-identification system:
7 ceiling cameras, 98 free-roaming cows, one 2.2-hour session, and **zero
identity annotations** during training. Rank-1 accuracy on a never-labelled
new camera rises from 0.503 (standard unsupervised baseline) to **0.945**,
within 2.4 points of a fully supervised model trained on the same backbone
(0.969).

The central measurement behind the design: cross-camera links mined at
52-60% precision collapse to ~17% pairwise precision once merged
transitively into clusters. The design rule follows: **pairwise, never
transitive** - every mined link trains exactly the pair it names and is
never chained into merged identities.

## Method in one paragraph

A four-stage "noise-bounded ensemble-distillation ladder", where every stage
limits the damage of pseudo-label errors: (1) self-training with
camera-aware proxies (a wrong merge corrupts one proxy, not a cluster
centroid); (2) a teacher built by averaging five independently seeded
models (cancels seed luck; mining in this space is 4x cleaner); (3)
frozen-label pairwise-link distillation into student models (no refresh, no
feedback loop; links act as confidence-weighted pairs); (4) heterogeneous
fusion of DINOv2 students with MegaDescriptor students (the two backbone
families fail on different animals and correct each other). A label-free
inference stack (768-d read-out, flip TTA, per-camera centering, whitening,
re-ranking, rank fusion) adds ~21 points without touching any weights.

## Data

The `2025Sep18` recording (124,145 body crops from 7 cameras) was collected
and preprocessed by **Phoenix Yu** (University of Bristol): SAM-3
segmentation, tracking by his modified gSORT, cross-camera identities
manually linked (used for evaluation only). The dataset is **not included**
in this repository; contact the data author. Pretrained backbones: DINOv2
(Meta AI), MegaDescriptor (BVRA). Everything downstream - tracklet
formation, free-signal mining, topology analysis, the training ladder, and
all evaluation - is this repository.

## Repository map

| Path | Contents |
|---|---|
| `cowreid/` | Library, imported by everything: manifest/tracklets/topology (data), pair mining + OT + constrained clustering (signals), encoder/losses/batching (training), splits/eval/ST-mask (evaluation) |
| `repro/` | **The 15 scripts that produce the headline numbers**, in run order - see `repro/README.md` |
| `common/` | 9 shared modules imported by the pipeline and the experiments: image cache and trainer core, IICS fine-tuning, label-free inference levers, ST evaluation helpers |
| `experiments/ablations/` | 33 scripts, one per claim: alternatives that were tried, including the negative results reported in Chapter 5 |
| `experiments/diagnostics/` | 18 audits that train nothing and establish one fact each (coat pattern, per-camera errors, pseudo-label precision) |
| `experiments/figure_scripts/` | 3 dissertation figure generators |
| `experiments/legacy/` | 3 superseded pre-ladder scripts, kept for provenance |
| `artifacts2/` | 100 archived result JSONs plus 5 control CSVs - every number in the dissertation traces to one of these; see `artifacts2/README.md` |
| `docs/reference/` | Per-stage fact sheets with file:line citations |
| `hpc/` | Slurm job scripts for the University of Bristol BluePebble cluster: the whole ladder as seven submissions, with walltime-safe resume |

Each folder carries its own README indexing every script in it with a one-line statement of
what question it answers and which archived JSON holds its result.

## Reproducing the pipeline

Roughly half a day on one 24 GB GPU. Run every command **from the repository root**: data and
artifact paths are relative to the working directory. Pretrained backbones download
automatically on first use.

```
# 0. place 2025Sep18.tar.gz + 2025Sep18.listing.txt in the repo root
pip install -r requirements.txt

# 1. signals and splits
python repro/build_phase1.py

# 1b. the uint8 image cache every trainer mmaps (15 GB on disk, CPU, ~1 h)
python common/vitb_unsup.py --build-cache --tar 2025Sep18.tar.gz

# 1c. the frozen ViT-S features stage 1 seeds its proxies from (9 MB, GPU, ~15 min)
python repro/make_vits_cache.py

# 2. stage 1: five CAP seeds (GPU, ~40 min each), seeds 0-4
python repro/vitb_unsup_cap.py --seed 0 --ckpt _vitb_cap_s0_ckpt.pt   # ... repeat for seeds 1-4

# 3. stage 2: ensemble teacher + frozen labels
python repro/eval_cap_ensemble.py
python repro/make_distill_labels.py

# 4. stage 3: students, 3 seeds each (deploy 10-12, hard cannot-link 16-18)
python repro/vitb_unsup_deploy.py  --seed 10 --ckpt _vitb_dep_s10_ckpt.pt
python repro/vitb_unsup_hardcl2.py --seed 16 --ckpt _vitb_hc2_s16_ckpt.pt

# 5. stage 4: MegaDescriptor students (seeds 40-42) + fusion
python repro/vitb_unsup_mega.py --seed 40 --ckpt _vitb_mega_s40_ckpt.pt
python repro/fuse_hetero.py
python repro/fuse_final.py

# 6. evaluation
python repro/eval_sweep.py ...
python repro/validate_protocols.py
```

Keep the seed numbers above: each one is written into its checkpoint filename, and
`fuse_final.py` locates every model's saved embeddings by that name.

**Reference environment.** Every reported number was produced on Windows 11 with Python
3.12.10, torch 2.11.0+cu128, timm 1.0.27, numpy 2.4.3, scikit-learn 1.8.0 and Pillow 12.1.1,
on a 24 GB RTX 5090 Laptop GPU. `requirements.txt` deliberately gives lower bounds rather
than pins, because a cluster's driver dictates which CUDA build of torch can be installed;
match the CUDA build to the machine and expect agreement within the seed variance rather
than to the last decimal.

Expect results within the reported seed variance (students +-0.03); every number can be
checked against the archived JSONs in `artifacts2/`. Bit-exact reproduction of the
dissertation's numbers requires the original checkpoints (48 checkpoints, ~51 GB - available
on request). All fusion and evaluation steps run on CPU from saved embeddings.

Nothing under `experiments/` is needed for the headline result. Those scripts are the
evidence for individual claims: one script per ablation, per negative result and per audit.

## Citation

Dissertation: *Pairwise, Not Transitive: Noise-Bounded Ensemble Distillation
for Label-Free Multi-Camera Cattle Re-Identification*, University of
Bristol, 2026. Related dataset lineage: Yu et al., *Computers and
Electronics in Agriculture* 2025 (MultiCamCows2024); Yu et al., arXiv
2602.15962 (DazzleCows).

## License

MIT (see LICENSE). Note: MegaDescriptor pretrained weights carry their own
licence terms - check the BVRA release before commercial use.
