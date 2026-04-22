"""SigLIP2 backbone wrapper with MiN PiNoise injection.

Normalization note
------------------
SigLIP2 was pretrained with mean=std=(0.5, 0.5, 0.5), which is identical to
MiN's data pipeline normalisation (mean=std=0.5).  No re-normalisation is
needed, unlike the DINOv2/v3 wrappers which must convert from (0.5,0.5,0.5)
inputs to ImageNet range.

Architecture note
-----------------
SigLIP2 SO400M/14 uses multi-head attention pooling (MAP via AttentionPoolLatent)
rather than a CLS token.  After the block stack and final LayerNorm, `attn_pool`
produces a single [B, embed_dim] representation.  The MAP head remains frozen;
only the PiNoise modules and per-block LayerNorms are trained by MiN.
"""

import torch
import torch.nn as nn
from backbones.ViT_MiN import PiNoise


SIGLIP2_CONFIGS = {
    'siglip2_so400m': {
        'model_name': 'ViT-SO400M-14-SigLIP2',
        'pretrained': 'webli',
        'fallbacks': [
            ('ViT-SO400M-14-SigLIP', 'webli'),
        ],
    },
}


class SigLIP2MiNWrapper(nn.Module):
    """open_clip SigLIP2 backbone with per-block PiNoise injection.

    The SigLIP2 weights are frozen; only the noise_maker modules and the
    per-block LayerNorms (.norm1, .norm2) and final .norm are unfrozen during
    MiN training via init_unfreeze().
    """

    def __init__(self, trunk, embed_dim: int, depth: int, hidden_dim: int):
        super().__init__()
        self.trunk = trunk

        # Expose the attributes accessed by MiNbaseNet.init_unfreeze() and update_noise()
        self.blocks = trunk.blocks   # nn.ModuleList; each has .norm1 and .norm2
        self.norm = trunk.norm       # final LayerNorm

        self.noise_maker = nn.Sequential(*[
            PiNoise(embed_dim, embed_dim, hidden_dim) for _ in range(depth)
        ])

        self.out_dim = embed_dim
        self.layer_num = depth

    def forward(self, x: torch.Tensor, new_forward: bool = False) -> torch.Tensor:
        # No re-normalisation: SigLIP2 and MiN both use mean=std=(0.5, 0.5, 0.5)
        x = self.trunk.patch_embed(x)   # [B, N, D]
        x = self.trunk._pos_embed(x)    # [B, N, D]  (no CLS token in SO400M MAP models)

        for i, block in enumerate(self.trunk.blocks):
            x = block(x)
            if new_forward:
                x = self.noise_maker[i].forward_new(x)
            else:
                x = self.noise_maker[i](x)

        x = self.trunk.norm(x)          # [B, N, D]
        x = self.trunk.attn_pool(x)     # [B, D]  MAP pooling → final feature
        return x


def load_siglip2_min(variant: str, hidden_dim: int) -> SigLIP2MiNWrapper:
    """Load SigLIP2 from open_clip and wrap it with MiN PiNoise injection.

    Requires open_clip >= 2.24 (available when called from the compcont-bench
    harness via MinTaskLearner).  Tries the primary weights then falls back.
    """
    if variant not in SIGLIP2_CONFIGS:
        raise ValueError(
            f"Unknown SigLIP2 variant '{variant}'. Available: {list(SIGLIP2_CONFIGS)}"
        )
    cfg = SIGLIP2_CONFIGS[variant]

    import open_clip  # available in compcont-bench's Python process

    loaded = False
    pairs = [(cfg['model_name'], cfg['pretrained'])] + cfg['fallbacks']
    for mn, pt in pairs:
        try:
            clip_model, _, _ = open_clip.create_model_and_transforms(mn, pretrained=pt)
            loaded = True
            break
        except Exception:
            continue

    if not loaded:
        raise RuntimeError(
            f"Could not load SigLIP2 variant '{variant}'. "
            "Check open_clip >= 2.24 and that weights are downloadable."
        )

    trunk = clip_model.visual.trunk
    for p in trunk.parameters():
        p.requires_grad = False

    embed_dim = trunk.embed_dim
    depth = len(trunk.blocks)

    return SigLIP2MiNWrapper(trunk, embed_dim, depth, hidden_dim)
