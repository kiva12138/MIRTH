"""
Test the MIRTH model defined in models/vla_model.py.
"""

import torch

from config.config_vla import (
    ACTION_DIM,
    ACTION_REASON_TOKEN_BEGIN_IDX,
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from models.vla_model import MIRTH, MIRTHConfig


def build_synthetic_batch(B, history_len, num_cameras, num_reason_token, prompt_len, device):
    """Construct a minimally well-formed input batch.

    Layout of each input_ids row (length = prompt_len + num_reason_token + NUM_ACTIONS_CHUNK):
        [random prompt tokens] [reason tokens] [action tokens]
    Labels mirror the input_ids; reason/action positions get their special IDs so
    that `get_action_tokens_mask` and `get_reasoning_tokens_mask` pick them up.
    """

    seq_len = prompt_len + num_reason_token + NUM_ACTIONS_CHUNK
    input_ids = torch.randint(low=10, high=1000, size=(B, seq_len), dtype=torch.long, device=device)
    input_ids[:, 0] = 1  # BOS

    reason_slice = slice(prompt_len, prompt_len + num_reason_token)
    action_slice = slice(prompt_len + num_reason_token, seq_len)
    input_ids[:, reason_slice] = torch.arange(
        ACTION_REASON_TOKEN_BEGIN_IDX, ACTION_REASON_TOKEN_BEGIN_IDX + num_reason_token,
        device=device,
    )
    input_ids[:, action_slice] = torch.arange(
        ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTIONS_CHUNK,
        device=device,
    )

    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    C, H, W = 3, 224, 224
    pixel_values = {
        'dino': torch.randn(B, num_cameras, C, H, W, device=device),
        'siglip': torch.randn(B, num_cameras, C, H, W, device=device),
    }
    pixel_values_history = {
        'dino': torch.randn(B, num_cameras, history_len, C, H, W, device=device),
        'siglip': torch.randn(B, num_cameras, history_len, C, H, W, device=device),
    }

    current_action = torch.randn(B, ACTION_DIM, device=device)
    current_action_chunk = torch.randn(B, NUM_ACTIONS_CHUNK, ACTION_DIM, device=device)
    history_actions = torch.randn(B, history_len, ACTION_DIM, device=device)

    proprio = torch.randn(B, PROPRIO_DIM, device=device)
    proprio_history = torch.randn(B, history_len, PROPRIO_DIM, device=device)

    # pad_mask covers (history_len + 1) — all real for this synthetic batch except
    # the very first frame, to exercise the masking branch.
    pad_mask = torch.ones(B, history_len + 1, dtype=torch.bool, device=device)
    pad_mask[:, 0] = False

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        current_action=current_action,
        current_action_chunk=current_action_chunk,
        history_actions=history_actions,
        pixel_values=pixel_values,
        pixel_values_history=pixel_values_history,
        proprio=proprio,
        proprio_history=proprio_history,
        pad_mask=pad_mask,
    )


def test_mirth_forward():
    print("=" * 60)
    print("Testing MIRTH.forward")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    config = MIRTHConfig(
        pretrained_vla_path="/mnt/data1/OpenVLA/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt",
        num_images_in_input=2,
        use_proprio=True,
        hf_token="/path/to/hf_token",
        action_token_type="one_for_action_step",
        use_timestamp=False,
        action_biattnn=False,
        use_vision_memory_hub=True,
        use_proprio_memory_hub=True,
        use_action_memory_hub=False,
        mb_prefix_type="union",
        long_memory_scale_number=4,
        short_memory_length=4,
        tau=1.0,
        beta_min=0.01,
        beta_max=0.3,
        gamma=0.2,
        lmbd=0.2,
        bias=1.0,
        use_reason_token=True,
        num_reason_token=4,
        reason_hidden=128,
        reason_p_drop=0.0,
        reason_out_scale=1.0,
        use_contrastive_loss=False,
    )

    B, history_len, num_cameras = 2, 20, config.num_images_in_input
    batch = build_synthetic_batch(
        B=B,
        history_len=history_len,
        num_cameras=num_cameras,
        num_reason_token=config.num_reason_token,
        prompt_len=8,
        device=device,
    )

    print("-" * 60)
    print("Synthetic batch contents:")
    print("-" * 60)
    for k, v in batch.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}[{kk}]: shape={tuple(vv.shape)}, dtype={vv.dtype}, "
                      f"min={vv.min().item():.3f}, max={vv.max().item():.3f}, "
                      f"mean={vv.float().mean().item():.3f}")
        elif v.dtype == torch.bool:
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"num_true={v.sum().item()}/{v.numel()}")
            print(f"    values: {v.tolist()}")
        elif v.dtype in (torch.long, torch.int32, torch.int64):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item()}, max={v.max().item()}")
            print(f"    row 0: {v[0].tolist()}")
        else:
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item():.3f}, max={v.max().item():.3f}, "
                  f"mean={v.mean().item():.3f}")
    print("-" * 60)
    print("Special-token positions in row 0 of input_ids:")
    from utils.train_utiils import get_action_tokens_mask, get_reasoning_tokens_mask
    action_mask = get_action_tokens_mask(batch['labels'])
    reason_mask = get_reasoning_tokens_mask(batch['labels'])
    print(f"  action token mask row 0: {action_mask[0].tolist()}  (count={action_mask[0].sum().item()})")
    print(f"  reason token mask row 0: {reason_mask[0].tolist()}  (count={reason_mask[0].sum().item()})")
    print("-" * 60)

    model = MIRTH(config).to(device)
    model.eval()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = model(**batch)
    loss = out.loss
    loss.backward()  # check that backward pass works without NaNs/Infs

    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.actions.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM), out.actions.shape
    print("  loss:", float(out.loss))
    print("  actions shape:", tuple(out.actions.shape))
    print("  logits shape:", tuple(out.logits.shape))
    print("  metrics:", out.metrics)


def test_mirth_freeze_backbones():
    """Smoke-check the freeze schedule used in `finetune_ddp.py`."""
    print("=" * 60)
    print("Testing MIRTH.freeze_backbones('lvp')")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MIRTHConfig(
        pretrained_vla_path="/mnt/data1/OpenVLA/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt",
        num_images_in_input=2,
        use_proprio=True,
        hf_token="/path/to/hf_token",
        action_token_type="one_for_action_step",
        use_vision_memory_hub=True,
        use_proprio_memory_hub=True,
        use_action_memory_hub=False,
        mb_prefix_type="union",
        use_reason_token=True,
        num_reason_token=4,
    )
    model = MIRTH(config).to(device)
    model.freeze_backbones("lvp")

    # Backbones frozen
    assert not any(p.requires_grad for p in model.vision_backbone.vision_backbone.parameters())
    assert not any(p.requires_grad for p in model.llm_backbone.parameters())
    assert not any(p.requires_grad for p in model.projector.parameters())
    # Memory hubs and action head trainable
    assert all(p.requires_grad for p in model.action_head.parameters())
    assert all(p.requires_grad for p in model.vision_memory_hub.parameters())
    assert all(p.requires_grad for p in model.proprio_memory_hub.parameters())

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_total/1e6:.2f}M, trainable: {n_trainable/1e6:.2f}M")


if __name__ == "__main__":
    test_mirth_forward()
    test_mirth_freeze_backbones()
    print("=" * 60)
    print("All MIRTH tests passed.")
