# The cowreid package

The foundation layer of the library: data model, free-signal mining, training
components and evaluation. Generic module names (`cluster`, `eval`, `features`)
are deliberately kept inside this package namespace so they never collide with
third-party libraries. Two design-history documents cover the early phases:
[README_phase1.md](README_phase1.md) (data model and free signals) and
[README_phase2.md](README_phase2.md) (training scaffolding).

All code in this package is this repository's own implementation. Where a
module realises a published method, the origin is listed below; full citations
are in the dissertation's bibliography.

| Module | Role | Method origin |
|---|---|---|
| `manifest.py` | Parse the recording listing into frames/crops | ours |
| `tracklets.py` | Assemble per-camera tracklets (max-gap rule) | ours |
| `topology.py` | Camera-overlap topology from co-occurrence | ours |
| `pair_miner.py` | Free-signal pair mining, constrained union-find | ours |
| `cluster.py` | Mutual-kNN clustering under cannot-link constraints | ours; cannot-link constraints are classic constrained clustering |
| `cajaccard.py` | Camera-aware Jaccard re-ranking distance | CA-Jaccard (Chen et al., 2024), built on k-reciprocal re-ranking (Zhong et al., 2017) |
| `crossview_ot.py` | Crop-bag must-link mining via optimal transport with a reject bin | mining design ours; OT solver per Cuturi (2013) |
| `sinkhorn.py` | Entropic optimal-transport solver | Sinkhorn distances (Cuturi, 2013) |
| `encoder.py` | DINOv2 backbone wrapper, partial unfreeze, temporal attention pooling | DINOv2 (Oquab et al., 2024); pooling ours |
| `losses.py` | Contrastive, cluster-memory and cannot-link objectives | InfoNCE-style contrastive (Chen et al., 2020); cluster memory per Cluster Contrast (Dai et al., 2022) and hybrid memory (Ge et al., 2020); cannot-link hinge ours |
| `iics.py` | Intra/inter-camera similarity training branches | IICS (Xuan & Zhang, 2021) |
| `batch.py` | Batch construction for the trainers | ours |
| `features.py` | Cached frozen features, colour histograms | ours |
| `splits.py` | Frozen evaluation splits | ours |
| `eval.py` | Rank-k / mAP scoring with same-camera junk removal | protocol conventions of Market-1501 (Zheng et al., 2015) |
| `st_inference.py` | Spatio-temporal impossibility mask at retrieval | rule design ours; ST-for-ReID lineage (Lv et al., 2018; Wang et al., 2019) |
| `io_utils.py` | CSV/JSON writers | ours |
| `remine.py` | Link re-mining in student spaces | ours |
