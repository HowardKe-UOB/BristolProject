"""Video Re-ID encoder: per-frame DINOv2 backbone + temporal pooling + projection.

A tracklet clip ``(B, T, C, H, W)`` -> a clip embedding ``(B, proj_dim)``. Single
frames ``(B, C, H, W)`` are accepted (treated as T=1). The backbone can be frozen for
warm-up / mining and unfrozen for fine-tuning. Swap ``model_name`` for any timm ViT;
the default is DINOv2 ViT-S/14.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Backbone(nn.Module):
    """Per-frame feature extractor (timm). Outputs (B, embed_dim)."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m",
                 pretrained: bool = True, freeze: bool = False):
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.num_features
        self.frozen = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def unfreeze_last(self, n_blocks: int):
        """Partial fine-tune: train only the last ``n_blocks`` transformer blocks +
        final norm (LoRA-free). Lower blocks stay frozen but still run in the graph."""
        self.frozen = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        for blk in self.model.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if hasattr(self.model, "norm"):
            for p in self.model.norm.parameters():
                p.requires_grad_(True)
        return self

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class TemporalPool(nn.Module):
    """Pool a sequence (B, T, D) -> (B, D) by mean or lightweight attention."""

    def __init__(self, dim: int, mode: str = "attn"):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.score = nn.Linear(dim, 1)
        elif mode != "mean":
            raise ValueError(f"unknown pool mode {mode!r}")

    def forward(self, feats):  # (B, T, D)
        if self.mode == "mean":
            return feats.mean(dim=1)
        w = torch.softmax(self.score(feats).squeeze(-1), dim=1)  # (B, T)
        return (feats * w.unsqueeze(-1)).sum(dim=1)


class VideoReIDEncoder(nn.Module):
    """DINOv2 + temporal pool + projection head.

    forward(clips) -> {"feat": (B, embed_dim) normalised backbone feature,
                       "embed": (B, proj_dim) normalised projected embedding}
    Use ``embed`` for the contrastive / cluster / cannot-link losses.
    """

    def __init__(self, backbone: DinoV2Backbone, proj_dim: int = 256,
                 pool: str = "attn"):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.pool = TemporalPool(self.embed_dim, pool)
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, proj_dim),
        )

    def forward(self, clips):
        if clips.dim() == 4:           # (B, C, H, W) -> T=1
            clips = clips.unsqueeze(1)
        B, T = clips.shape[:2]
        frames = clips.flatten(0, 1)                       # (B*T, C, H, W)
        f = self.backbone(frames).view(B, T, -1)           # (B, T, D)
        feat = self.pool(f)                                # (B, D)
        embed = self.proj(feat)                            # (B, proj_dim)
        return {"feat": F.normalize(feat, dim=1),
                "embed": F.normalize(embed, dim=1)}

    @torch.no_grad()
    def embed_clip(self, clips):
        return self(clips)["embed"]
