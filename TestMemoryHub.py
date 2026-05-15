import torch

from models.memory_hub import (
    VisionMemoryHubForTraining,
    ProprioMemoryHubForTraining,
)
from models.vision_encoders import get_Prism_7B_DINOSigLIP_224px_backbone_and_transform
from models.infuse_wrapper import InfusedDinoSigLIPViTBackbone


def test_vision_memory_hub():
    print("=" * 60)
    print("Testing VisionMemoryHubForTraining")
    print("=" * 60)

    torch.manual_seed(0)
    B, num_cameras, history_len, C, H, W = 4, 2, 9, 3, 224, 224
    
    vision_encoder, _ = get_Prism_7B_DINOSigLIP_224px_backbone_and_transform()
    infused_vision_encoder = InfusedDinoSigLIPViTBackbone(vision_encoder, infuse=True, infuse_layers_ratio=0.25).cuda()

    memory_hub = VisionMemoryHubForTraining(
        hidden_dim=vision_encoder.embed_dim,
        long_memory_scale_number=4,
        short_memory_length=3,
    ).cuda()

    pixel_values = {
        'dino': torch.randn(B, num_cameras, C, H, W).cuda(),
        'siglip': torch.randn(B, num_cameras, C, H, W).cuda(),
    }
    pixel_values_history = {
        'dino': torch.randn(B, num_cameras, history_len, C, H, W).cuda(),
        'siglip': torch.randn(B, num_cameras, history_len, C, H, W).cuda(),
    }
    pad_mask = torch.ones(B, history_len + 1, dtype=torch.bool).cuda()
    # Mark a couple of early frames as padding for the first sample.
    pad_mask[0, :2] = False

    out = memory_hub(pixel_values, pixel_values_history, infused_vision_encoder, pad_mask)
    expected_N = num_cameras * vision_encoder.num_patches
    assert out['workspace'].shape == (B, expected_N, vision_encoder.embed_dim), out['workspace'].shape
    assert out['short'].shape == (B, expected_N, vision_encoder.embed_dim), out['short'].shape
    assert out['mem'].shape == (B, expected_N, vision_encoder.embed_dim), out['mem'].shape
    print("  forward output shapes OK:",
          {k: tuple(v.shape) for k, v in out.items()})


def test_proprio_memory_hub():
    print("=" * 60)
    print("Testing ProprioMemoryHubForTraining")
    print("=" * 60)

    torch.manual_seed(1)
    B, history_len, hidden_dim = 2, 6, 12
    hub = ProprioMemoryHubForTraining(
        hidden_dim=hidden_dim,
        long_memory_scale_number=4,
        short_memory_length=3,
    ).cuda()

    proprio = torch.randn(B, hidden_dim).cuda()
    proprio_history = torch.randn(B, history_len, hidden_dim).cuda()
    pad_mask = torch.ones(B, history_len + 1, dtype=torch.bool).cuda()
    pad_mask[1, :1] = False

    out = hub(proprio, proprio_history, pad_mask)
    assert out['workspace'].shape == (B, hidden_dim), out['workspace'].shape
    assert out['short'].shape == (B, hidden_dim), out['short'].shape
    assert out['mem'].shape == (B, hidden_dim), out['mem'].shape
    print("  forward output shapes OK:",
          {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    test_vision_memory_hub()
    test_proprio_memory_hub()
    print("=" * 60)
    print("All memory hub tests passed.")
