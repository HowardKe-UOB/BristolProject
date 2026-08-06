"""Phase-2 training scaffolding: assemble encoder + multi-task objective and run a
training step. Run directly for a self-contained smoke test (CPU, random weights,
no downloads) that exercises every component and a backward pass.

Training loop sketch
--------------------
    encoder   = VideoReIDEncoder(DinoV2Backbone(pretrained=True, freeze=warmup))
    objective = build_objective(dim=encoder.proj_dim=256, num_clusters=K)
    # each epoch: refresh pseudo-labels with current encoder, resize the cluster
    # memory to the new cluster count, then iterate batches
    labels = refresh_pseudo_labels(encoder, tracklets, loader, transform, topology)
    memory.reset(ClusterAssigner.num_clusters(labels))            # K may change
    builder = RelationBatchBuilder(index, labels, build_cannot_link(tracklets, topo))
    for step in range(steps):
        batch = builder.sample(num_pseudo_ids=P, per_id=K)
        clips = load_clips(batch, loader, transform, device)
        comp  = train_step(encoder, objective, clips, batch, optimizer, device)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import numpy as np
import torch

from cowreid.batch import RelationBatch, RelationBatchBuilder, load_clips
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.losses import (CannotLinkLoss, ClusterContrastLoss, ClusterMemory,
                            MultiTaskSSLObjective, NegativeAwareContrastiveLoss)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_objective(dim: int, num_clusters: int, temperature: float = 0.07,
                    cluster_temp: float = 0.05, momentum: float = 0.2,
                    w_contrastive: float = 1.0, w_cluster: float = 1.0,
                    w_cannotlink: float = 0.5, hard_negative_weight: float = 2.0
                    ) -> tuple[MultiTaskSSLObjective, ClusterMemory]:
    memory = ClusterMemory(dim, num_clusters, temperature=cluster_temp, momentum=momentum)
    obj = MultiTaskSSLObjective(
        contrastive=NegativeAwareContrastiveLoss(temperature),
        cluster=ClusterContrastLoss(memory),
        cannotlink=CannotLinkLoss(margin=0.0),
        w_contrastive=w_contrastive, w_cluster=w_cluster,
        w_cannotlink=w_cannotlink, hard_negative_weight=hard_negative_weight)
    return obj, memory


def train_step(encoder, objective, clips, batch: RelationBatch, optimizer,
               device: str = "cpu") -> dict:
    encoder.train()
    embed = encoder(clips.to(device))["embed"]
    total, comp = objective(
        embed,
        positive_mask=batch.positive_mask.to(device),
        hard_negative_mask=batch.hard_negative_mask.to(device),
        forbid_negative_mask=batch.forbid_negative_mask.to(device),
        cluster_labels=batch.pseudo_labels.to(device),
        cannot_link_pairs=batch.cannot_link_pairs.to(device))
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return {k: float(v) for k, v in comp.items()}


@torch.no_grad()
def compute_tracklet_embeddings(encoder, tracklets, loader, transform,
                                device: str = "cpu", frames_per_tracklet: int = 4
                                ) -> tuple[list[str], np.ndarray]:
    """Mean clip embedding per tracklet -- input to :class:`ClusterAssigner`."""
    encoder.eval()
    ids, vecs = [], []
    for tr in tracklets:
        paths = tr.paths
        idx = np.linspace(0, len(paths) - 1, min(frames_per_tracklet, len(paths))).astype(int)
        frames = torch.stack([transform(loader.load(paths[i])) for i in idx]).unsqueeze(0)
        vecs.append(encoder(frames.to(device))["embed"][0].cpu().numpy())
        ids.append(tr.tracklet_id)
    return ids, np.stack(vecs)


def refresh_pseudo_labels(encoder, tracklets, loader, transform, topology,
                          device: str = "cpu", sim_threshold: float = 0.6,
                          overlap_threshold: float = 0.02) -> dict[str, int]:
    ids, feats = compute_tracklet_embeddings(encoder, tracklets, loader, transform, device)
    cl = build_cannot_link(tracklets, topology, overlap_threshold)
    return ClusterAssigner(sim_threshold=sim_threshold).assign(ids, feats, cl)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _smoke():
    torch.manual_seed(0)
    device = "cpu"
    print("== 1. losses on synthetic embeddings ==")
    B, D = 12, 32
    z = torch.randn(B, D, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1])
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (labels.unsqueeze(0) >= 0)
    hard = torch.zeros(B, B, dtype=torch.bool); hard[0, 6] = hard[6, 0] = True
    cl_pairs = torch.tensor([[0, 6], [3, 8]])
    obj, mem = build_objective(dim=D, num_clusters=3)
    total, comp = obj(z, positive_mask=pos, hard_negative_mask=hard,
                      cluster_labels=labels, cannot_link_pairs=cl_pairs)
    total.backward()
    assert torch.isfinite(total) and z.grad is not None
    print("   loss components:", {k: round(v.item(), 4) for k, v in comp.items()})

    print("== 2. ClusterAssigner respects cannot-link ==")
    f = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=np.float32)
    ids = ["a", "b", "c", "d"]
    free = ClusterAssigner(sim_threshold=0.8).assign(ids, f)
    forced = ClusterAssigner(sim_threshold=0.8).assign(ids, f, cannot_link={frozenset(("a", "b"))})
    print("   no constraint:", free, "-> clusters:", ClusterAssigner.num_clusters(free))
    print("   cannot-link a-b:", forced)
    assert free["a"] == free["b"] and forced["a"] != forced["b"]

    print("== 3. encoder forward + full backward (vit_tiny, random weights) ==")
    from cowreid.encoder import DinoV2Backbone, VideoReIDEncoder
    enc = VideoReIDEncoder(DinoV2Backbone("vit_tiny_patch16_224", pretrained=False),
                           proj_dim=64, pool="attn").to(device)
    Bt, T = 6, 2
    clips = torch.randn(Bt, T, 3, 224, 224)
    out = enc(clips)
    assert out["embed"].shape == (Bt, 64)
    print("   embed shape:", tuple(out["embed"].shape), "| feat:", tuple(out["feat"].shape))
    obj2, _ = build_objective(dim=64, num_clusters=2)
    labs = torch.tensor([0, 0, 1, 1, -1, -1])
    posm = (labs.unsqueeze(0) == labs.unsqueeze(1)) & (labs.unsqueeze(0) >= 0)
    opt = torch.optim.SGD(enc.parameters(), lr=0.01)
    fake = RelationBatch(["t"] * Bt, [[]], ["c"] * Bt, labs, posm,
                         torch.zeros(Bt, Bt, dtype=torch.bool),
                         torch.zeros(Bt, Bt, dtype=torch.bool),
                         torch.tensor([[0, 2]]))
    comp = train_step(enc, obj2, clips, fake, opt, device)
    print("   train_step components:", {k: round(v, 4) for k, v in comp.items()})

    print("== 4. batch builder + cannot-link on real Phase-1 tracklets ==")
    from cowreid import Manifest, build_tracklets, CameraTopology, TrackletIndex
    m = Manifest.from_listing_file("2025Sep18.listing.txt")
    tr = build_tracklets(m, max_gap_s=2)
    topo = CameraTopology.from_gt(m)
    cl = build_cannot_link(tr, topo, 0.02)
    print(f"   tracklets={len(tr)}  cannot-link tracklet-pairs={len(cl)}")
    # fake pseudo-labels = gt for a quick mask sanity check
    idx = TrackletIndex(tr)
    labels = {t.tracklet_id: int(t.gt_label) for t in tr}
    builder = RelationBatchBuilder(idx, labels, cl, clip_len=4, seed=0)
    b = builder.sample(num_pseudo_ids=8, per_id=2)
    print(f"   batch B={len(b.tracklet_ids)}  positives={int(b.positive_mask.sum())}  "
          f"hard_negs={int(b.hard_negative_mask.sum())}  cl_pairs={b.cannot_link_pairs.shape[0]}")
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    _smoke()
