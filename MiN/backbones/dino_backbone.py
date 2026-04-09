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

Normalization note
------------------
The MiN data pipeline normalises images with mean=std=(0.5, 0.5, 0.5), which
matches the original timm ViT-B/16 pretraining.  DINOv2 and DINOv3 were both
pretrained with ImageNet normalisation (mean=(0.485,0.456,0.406),
std=(0.229,0.224,0.225)).  Using the wrong input statistics causes the per-block
LayerNorms fine-tuned by ``init_unfreeze`` to diverge, producing gradient
explosions after ~100-700 gradient steps depending on dataset size.  Both
wrappers therefore apply a fixed affine re-normalisation at the start of
``forward`` to convert the incoming (0.5, 0.5, 0.5) images to the expected
ImageNet range before feeding them to the backbone.
"""

import torch
import torch.nn as nn
from backbones.ViT_MiN import PiNoise

# ImageNet normalisation constants (what DINOv2/v3 were pretrained on)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# MiN data pipeline normalisation: mean = std = 0.5.
# The pre-processed pixel value is  x_in = (p - 0.5) / 0.5,  p in [0, 1].
# We want                            x_out = (p - mu) / sigma.
# Combined:  x_out = x_in * (0.5 / sigma) + (0.5 - mu) / sigma.
_RENORM_SCALE = torch.tensor(
    [0.5 / s for s in _IMAGENET_STD], dtype=torch.float32
).view(1, 3, 1, 1)
_RENORM_SHIFT = torch.tensor(
    [(0.5 - m) / s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)], dtype=torch.float32
).view(1, 3, 1, 1)


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

        # Fixed re-normalisation: convert MiN's (0.5,0.5,0.5) inputs to ImageNet range.
        self.register_buffer('renorm_scale', _RENORM_SCALE.clone())
        self.register_buffer('renorm_shift', _RENORM_SHIFT.clone())

    def forward(self, x: torch.Tensor, new_forward: bool = False) -> torch.Tensor:
        # Re-normalise from MiN's (mean=0.5, std=0.5) to ImageNet (mean≈0.485, std≈0.229)
        x = x * self.renorm_scale + self.renorm_shift
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

    When ``use_rope`` is True (DINOv3 ViT layers), the layer must receive
    ``position_embeddings`` from RoPE; forward then passes them through.
    """

    def __init__(self, layer, *, use_rope: bool = False):
        super().__init__()
        self.layer = layer
        self.use_rope = use_rope
        if hasattr(layer, 'norm1'):
            self.norm1 = layer.norm1
            self.norm2 = layer.norm2
        else:
            # Standard HF ViTLayer naming
            self.norm1 = layer.layernorm_before
            self.norm2 = layer.layernorm_after

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.use_rope:
            return self.layer(x, position_embeddings=position_embeddings)
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


def _dinov3_transformer_layers(hf_model: nn.Module) -> nn.ModuleList:
    """Resolve the transformer block stack for DINOv3 (HF layout varies by version)."""
    enc = getattr(hf_model, 'encoder', None)
    if enc is not None and hasattr(enc, 'layer'):
        return enc.layer
    inner = getattr(hf_model, 'model', None)
    if inner is not None and hasattr(inner, 'layer'):
        return inner.layer
    raise AttributeError(
        'Expected hf_model.encoder.layer or hf_model.model.layer for DINOv3'
    )


def _dinov3_final_norm(hf_model: nn.Module) -> nn.Module:
    """Final LayerNorm before pooling (``layernorm`` or ``norm`` depending on version)."""
    if hasattr(hf_model, 'layernorm'):
        return hf_model.layernorm
    if hasattr(hf_model, 'norm'):
        return hf_model.norm
    raise AttributeError('Expected hf_model.layernorm or hf_model.norm for DINOv3')


class DINOv3MiNWrapper(nn.Module):
    """DINOv3 (HuggingFace) backbone with per-block PiNoise injection.

    The DINOv3 weights are frozen; only the noise_maker modules train.
    """

    def __init__(self, hf_model, embed_dim: int, depth: int, hidden_dim: int):
        super().__init__()
        self.hf_model = hf_model

        layers = _dinov3_transformer_layers(hf_model)
        # Wrap encoder layers so each exposes .norm1 and .norm2
        self.blocks = nn.ModuleList([
            _HFBlockAdapter(layer, use_rope=True)
            for layer in layers
        ])
        self.norm = _dinov3_final_norm(hf_model)

        self.noise_maker = nn.Sequential(*[
            PiNoise(embed_dim, embed_dim, hidden_dim) for _ in range(depth)
        ])

        self.out_dim = embed_dim
        self.layer_num = depth

        # Fixed re-normalisation: convert MiN's (0.5,0.5,0.5) inputs to ImageNet range.
        self.register_buffer('renorm_scale', _RENORM_SCALE.clone())
        self.register_buffer('renorm_shift', _RENORM_SHIFT.clone())

    def forward(self, x: torch.Tensor, new_forward: bool = False) -> torch.Tensor:
        # Re-normalise from MiN's (mean=0.5, std=0.5) to ImageNet (mean≈0.485, std≈0.229)
        x = x * self.renorm_scale + self.renorm_shift
        # Match DINOv3ViTModel.forward: embeddings + RoPE, then blocks + final norm.
        pixel_values = x.to(self.hf_model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.hf_model.embeddings(pixel_values)
        position_embeddings = self.hf_model.rope_embeddings(pixel_values)

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, position_embeddings=position_embeddings
            )
            if new_forward:
                hidden_states = self.noise_maker[i].forward_new(hidden_states)
            else:
                hidden_states = self.noise_maker[i](hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states[:, 0]                      # CLS token → [B, D]


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
    depth = len(_dinov3_transformer_layers(hf_model))
    embed_dim = cfg['embed_dim'] or hf_model.config.hidden_size

    return DINOv3MiNWrapper(hf_model, embed_dim, depth, hidden_dim)

