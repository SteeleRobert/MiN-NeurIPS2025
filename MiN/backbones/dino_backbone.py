"""DINOv2 and DINOv3 backbone wrappers with MiN PiNoise injection.

Usage (via backbone_type in model config):
    "dinov2_vits14"  / "dinov2_vitb14"  / "dinov2_vitl14"  -- torch.hub
    "dinov3_vits16"  / "dinov3_vitb16"  / "dinov3_vitl16"  -- HuggingFace

Both wrappers expose the same interface as VisionTransformer in ViT_MiN.py:
    .blocks          -- nn.ModuleList, each element has .norm1 and .norm2
    .norm            -- final LayerNorm
    .noise_maker     -- nn.Sequential of PiNoise modules (one per block)
    .out_dim         -- embedding dimension
    .layer_num       -- number of transformer blocks
    forward(x, new_forward=False) -> [B, embed_dim]  (CLS token)
"""

import torch
import torch.nn as nn
from backbones.ViT_MiN import PiNoise


# ── DINOv2 (torch.hub) ───────────────────────────────────────────────────────

DINOV2_CONFIGS = {
    'dinov2_vits14':     {'embed_dim': 384, 'depth': 12},
    'dinov2_vitb14':     {'embed_dim': 768, 'depth': 12},
    'dinov2_vitb14_reg': {'embed_dim': 768, 'depth': 12},
}


class DINOv2MiNWrapper(nn.Module):
    """DINOv2 (torch.hub) backbone with per-block PiNoise injection.

    The DINOv2 weights are frozen; only the noise_maker modules train.
    """

    def __init__(self, dino_model, embed_dim: int, depth: int, hidden_dim: int):
        super().__init__()
        self.dino = dino_model

        # Expose attributes that MiNbaseNet.init_unfreeze() / update_noise() access
        self.blocks = dino_model.blocks   # nn.ModuleList; each block has .norm1, .norm2
        self.norm = dino_model.norm       # final LayerNorm

        self.noise_maker = nn.Sequential(*[
            PiNoise(embed_dim, embed_dim, hidden_dim) for _ in range(depth)
        ])

        self.out_dim = embed_dim
        self.layer_num = depth

    def forward(self, x: torch.Tensor, new_forward: bool = False) -> torch.Tensor:
        # Tokenise + add cls token + positional embeddings
        x = self.dino.prepare_tokens_with_masks(x, None)   # [B, 1+N, D]

        for i, block in enumerate(self.dino.blocks):
            x = block(x)                                    # [B, 1+N, D]
            if new_forward:
                x = self.noise_maker[i].forward_new(x)
            else:
                x = self.noise_maker[i](x)

        x = self.dino.norm(x)
        return x[:, 0]                                      # CLS token → [B, D]


def load_dinov2_min(variant: str, hidden_dim: int) -> DINOv2MiNWrapper:
    """Load DINOv2 from torch.hub and wrap it with MiN PiNoise injection."""
    if variant not in DINOV2_CONFIGS:
        raise ValueError(
            f"Unknown DINOv2 variant '{variant}'. Available: {list(DINOV2_CONFIGS)}"
        )
    cfg = DINOV2_CONFIGS[variant]
    dino = torch.hub.load('facebookresearch/dinov2:main', variant)
    for p in dino.parameters():
        p.requires_grad = False
    return DINOv2MiNWrapper(dino, cfg['embed_dim'], cfg['depth'], hidden_dim)


# ── DINOv3 (HuggingFace) ─────────────────────────────────────────────────────

DINOV3_CONFIGS = {
    'dinov3_vits16': {
        'hf_id': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'embed_dim': 384,
        'depth': 12,
    },
    'dinov3_vitb16': {
        'hf_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'embed_dim': 768,
        'depth': 12,
    },
    'dinov3_vitl16': {
        'hf_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'embed_dim': 1024,
        'depth': 24,
    },
    'dinov3_vith16': {
        'hf_id': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'embed_dim': None,   # resolved from model.config.hidden_size at load time
        'depth': None,       # resolved from len(model.encoder.layer) at load time
    },
    'dinov3_vit7b16': {
        'hf_id': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        'embed_dim': 4096,
        'depth': None,       # resolved from len(model.encoder.layer) at load time
    },
}


class _HFBlockAdapter(nn.Module):
    """Wrap one HuggingFace encoder layer to expose .norm1 / .norm2.

    Handles both Dinov2Layer (.norm1/.norm2) and ViTLayer
    (.layernorm_before/.layernorm_after).
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        if hasattr(layer, 'norm1'):
            self.norm1 = layer.norm1
            self.norm2 = layer.norm2
        else:
            # Standard HF ViTLayer naming
            self.norm1 = layer.layernorm_before
            self.norm2 = layer.layernorm_after

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF encoder layers return a tuple; first element is hidden states
        return self.layer(x)[0]


class DINOv3MiNWrapper(nn.Module):
    """DINOv3 (HuggingFace) backbone with per-block PiNoise injection.

    The DINOv3 weights are frozen; only the noise_maker modules train.
    """

    def __init__(self, hf_model, embed_dim: int, depth: int, hidden_dim: int):
        super().__init__()
        self.hf_model = hf_model

        # Wrap encoder layers so each exposes .norm1 and .norm2
        self.blocks = nn.ModuleList([
            _HFBlockAdapter(layer)
            for layer in hf_model.encoder.layer
        ])
        self.norm = hf_model.layernorm   # final LayerNorm

        self.noise_maker = nn.Sequential(*[
            PiNoise(embed_dim, embed_dim, hidden_dim) for _ in range(depth)
        ])

        self.out_dim = embed_dim
        self.layer_num = depth

    def forward(self, x: torch.Tensor, new_forward: bool = False) -> torch.Tensor:
        # Embed patches + prepend cls token + add positional embeddings
        x = self.hf_model.embeddings(x)    # [B, 1+N, D]

        for i, block in enumerate(self.blocks):
            x = block(x)                    # [B, 1+N, D]
            if new_forward:
                x = self.noise_maker[i].forward_new(x)
            else:
                x = self.noise_maker[i](x)

        x = self.hf_model.layernorm(x)
        return x[:, 0]                      # CLS token → [B, D]


def load_dinov3_min(variant: str, hidden_dim: int) -> DINOv3MiNWrapper:
    """Load DINOv3 from HuggingFace and wrap it with MiN PiNoise injection.

    Requires HuggingFace authentication for gated models:
        python -c "from huggingface_hub import login; login(token='TOKEN')"
    """
    from transformers import AutoModel

    if variant not in DINOV3_CONFIGS:
        raise ValueError(
            f"Unknown DINOv3 variant '{variant}'. Available: {list(DINOV3_CONFIGS)}"
        )
    cfg = DINOV3_CONFIGS[variant]
    hf_model = AutoModel.from_pretrained(cfg['hf_id'])
    for p in hf_model.parameters():
        p.requires_grad = False

    # Always resolve from the loaded model so None entries are handled correctly
    depth = len(hf_model.encoder.layer)
    embed_dim = cfg['embed_dim'] or hf_model.config.hidden_size

    return DINOv3MiNWrapper(hf_model, embed_dim, depth, hidden_dim)

