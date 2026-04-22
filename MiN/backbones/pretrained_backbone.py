from torch import nn
import timm
import torch
from backbones.ViT_MiN import VisionTransformer


def get_pretrained_backbone(args):
    name = args['backbone_type']
    hidden_dim = args.get('hidden_dim', 192)

    if name == "pretrained_vit_b16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        model.layer_num = 12
    elif name == "pretrained_vit_b16_224_in21k_min":
        model_f = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model = VisionTransformer(num_classes=0, weight_init='skip', args=args)
        model.load_state_dict(model_f.state_dict(), strict=False)
        model.out_dim = 768
        model.layer_num = 12
    elif name.startswith('dinov2_'):
        from backbones.dino_backbone import load_dinov2_min
        model = load_dinov2_min(name, hidden_dim)
    elif name.startswith('dinov3_'):
        from backbones.dino_backbone import load_dinov3_min
        model = load_dinov3_min(name, hidden_dim)
    elif name.startswith('siglip2_'):
        from backbones.siglip_backbone import load_siglip2_min
        model = load_siglip2_min(name, hidden_dim)
    else:
        raise ValueError(f"Unknown backbone_type: '{name}'")

    return model