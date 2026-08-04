# Phase 2 — Self-supervised training scaffolding (PyTorch)

Builds on Phase-1 signals. Direction (agreed): Tier-1 tracklet positives + appearance
clustering for cross-camera positives + heavy use of the temporal **hard negatives**
and **topology cannot-link**; revisit temporal cross-view positives via the re-mining
hook once the DINOv2 backbone is strong.

> Phase-2 modules import `torch`; they are intentionally **not** re-exported from
> `cowreid/__init__`, so the torch-free Phase-1 tooling still imports cleanly.
> Use `from cowreid.losses import ...`, `from cowreid.encoder import ...`, etc.

## Components

| Module | What it provides |
|---|---|
| `encoder.py` | `DinoV2Backbone` (timm ViT, freezable) → `TemporalPool` (mean/attn) → `VideoReIDEncoder`; clip `(B,T,C,H,W) → {"feat","embed"}` |
| `losses.py` | `NegativeAwareContrastiveLoss`, `ClusterMemory`+`ClusterContrastLoss`, `CannotLinkLoss`, `MultiTaskSSLObjective` |
| `cluster.py` | `build_cannot_link` (topology + time-overlap), `ClusterAssigner` (mutual-kNN + constrained union-find) |
| `batch.py` | `RelationBatchBuilder` (PK tracklet sampler → masks), `load_clips`, `build_transform` |
| `remine.py` | `EncoderFeatureExtractor` + `remine_overlap_positives` (mining ↔ training loop) |
| `train_phase2.py` | `build_objective`, `train_step`, `refresh_pseudo_labels`; `python train_phase2.py` runs a CPU smoke test |

## Loss interfaces (tensor-level, data-layer agnostic)

```python
# z: (B, D) embeddings;  *_mask: (B, B) bool;  cannot_link_pairs: (M, 2) long
total, comp = objective(
    embed,
    positive_mask=batch.positive_mask,            # same pseudo-id / same tracklet
    hard_negative_mask=batch.hard_negative_mask,  # temporal cannot-link (up-weighted)
    forbid_negative_mask=batch.forbid_negative_mask,  # false-negative suppression (opt)
    cluster_labels=batch.pseudo_labels,           # -1 = clustering outlier (ignored)
    cannot_link_pairs=batch.cannot_link_pairs)    # explicit topology penalty
```

* **NegativeAwareContrastiveLoss** — multi-positive InfoNCE; excludes forbidden pairs
  from negatives, injects/up-weights temporal hard negatives.
* **ClusterContrastLoss** — ClusterNCE against a momentum centroid memory; call
  `memory.reset(K)` whenever the cluster count changes after a refresh.
* **CannotLinkLoss** — hinge on cosine similarity of cannot-link pairs.

## Epoch loop

```python
labels  = refresh_pseudo_labels(encoder, tracklets, loader, transform, topology)
memory.reset(ClusterAssigner.num_clusters(labels))
builder = RelationBatchBuilder(index, labels, build_cannot_link(tracklets, topology))
for step in range(steps):
    batch = builder.sample(num_pseudo_ids=P, per_id=K)
    clips = load_clips(batch, loader, transform, device)
    comp  = train_step(encoder, objective, clips, batch, optimizer, device)

# periodically, once the encoder is strong:
new_pos = remine_overlap_positives(encoder, manifest, tracklets, topology,
                                   loader, transform, device)
```

## Notes
* Extract crops once (`extract_paths` / `tar -xzf`) and point `ImageLoader(root=...)`
  at the directory — far faster than reading from the tarball during training.
* Default backbone is DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, 518² input);
  the smoke test swaps in random-weight `vit_tiny_patch16_224` to avoid downloads.
* `cluster_labels=None` disables the cluster term for warm-up batches before the first
  assignment exists.
