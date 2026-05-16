import torch
from torch.utils.data import DataLoader

from config.config_vla import (
    ACTION_DIM,
    ACTION_REASON_TOKEN_BEGIN_IDX,
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from config.config_vlm import Prism_7B_DINOSigLIP_224px
from rlds_datasets.datasets import RLDSBatchTransform, RLDSDataset
from models.llm_llama2 import LLaMa2LLMBackbone
from models.vision_encoders import get_Prism_7B_DINOSigLIP_224px_backbone_and_transform
from utils.data_utils import PaddedCollatorForActionPrediction
from utils.train_utiils import get_action_tokens_mask, get_reasoning_tokens_mask


DATA_ROOT_DIR = "/media/sunhao/T7/LIBERO/modified_libero_rlds/"
DATASET_NAME = "libero_goal_no_noops"
HF_TOKEN = "your_huggingface_token_here"  # replace with your Hugging Face token if needed


def test_rlds_dataset():
    print("=" * 60)
    print("Building RLDSDataset + Collator")
    print("=" * 60)

    history_window_size = 20
    per_device_batch_size = 2
    num_reason_token = 4
    num_cameras = 2  # primary + wrist
    image_size = 224

    print("Loading vision backbone (for image_transform) and tokenizer ...")
    vision_backbone, image_transform = get_Prism_7B_DINOSigLIP_224px_backbone_and_transform()
    llm_backbone = LLaMa2LLMBackbone(
        config=Prism_7B_DINOSigLIP_224px(),
        hf_token=HF_TOKEN,
        use_flash_attention_2=False,
        load_pretrained=True,  # only need tokenizer/prompt_builder — skip the 7B weights
    )
    tokenizer = llm_backbone.tokenizer
    prompt_builder = llm_backbone.prompt_builder

    batch_transform = RLDSBatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder=prompt_builder,
        action_token_type="one_for_action_step",
        num_reason_tokens=num_reason_token,
        use_reason_token=True,
    )
    dataset = RLDSDataset(
        data_root_dir=DATA_ROOT_DIR,
        data_mix=DATASET_NAME,
        batch_transform=batch_transform,
        resize_resolution=(image_size, image_size),
        shuffle_buffer_size=1_000,
        load_proprio=True,
        load_camera_views=("primary", "wrist"),
        train=True,
        image_aug=True,
        history_window_size=history_window_size,
    )
    print(f"  dataset length (transitions): {len(dataset)}")
    print(f"  dataset statistics keys: {list(dataset.dataset_statistics.keys())}")

    collator = PaddedCollatorForActionPrediction(
        model_max_length=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    print("-" * 60)
    print("Pulling one collated batch ...")
    print("-" * 60)
    batch = next(iter(loader))

    print("Batch contents:")
    for k, v in batch.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}[{kk}]: shape={tuple(vv.shape)}, dtype={vv.dtype}, "
                      f"min={vv.min().item():.3f}, max={vv.max().item():.3f}, "
                      f"mean={vv.float().mean().item():.3f}")
        elif v is None:
            print(f"  {k}: None")
        elif v.dtype == torch.bool:
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"num_true={v.sum().item()}/{v.numel()}")
            print(f"    values: {v.tolist()}")
        elif v.dtype in (torch.long, torch.int32, torch.int64):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item()}, max={v.max().item()}")
            print(f"    row 0 (first 32 tokens): {v[0, :32].tolist()}")
            print(f"    row 0 (last 32 tokens):  {v[0, -32:].tolist()}")
        else:
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item():.3f}, max={v.max().item():.3f}, "
                  f"mean={v.mean().item():.3f}")

    print("-" * 60)
    print("Special token positions on row 0 of labels:")
    action_mask = get_action_tokens_mask(batch['labels'])
    reason_mask = get_reasoning_tokens_mask(batch['labels'])
    print(f"  action tokens count per row: {action_mask.sum(dim=1).tolist()} (expected {NUM_ACTIONS_CHUNK})")
    print(f"  reason tokens count per row: {reason_mask.sum(dim=1).tolist()} (expected {num_reason_token})")

    print("-" * 60)
    print("Shape sanity checks:")
    B = per_device_batch_size
    H_minus = history_window_size - 1
    expected = {
        'current_action': (B, ACTION_DIM),
        'current_action_chunk': (B, NUM_ACTIONS_CHUNK, ACTION_DIM),
        'history_actions': (B, H_minus, ACTION_DIM),
        'proprio': (B, PROPRIO_DIM),
        'proprio_history': (B, H_minus, PROPRIO_DIM),
        'pad_mask': (B, history_window_size),
    }
    for k, shp in expected.items():
        assert tuple(batch[k].shape) == shp, f"{k}: got {tuple(batch[k].shape)}, expected {shp}"
        print(f"  {k}: {tuple(batch[k].shape)} OK")

    assert batch['pixel_values']['dino'].shape == (B, num_cameras, 3, image_size, image_size)
    assert batch['pixel_values_history']['dino'].shape == (B, num_cameras, H_minus, 3, image_size, image_size)
    print(f"  pixel_values[dino]: {tuple(batch['pixel_values']['dino'].shape)} OK")
    print(f"  pixel_values_history[dino]: {tuple(batch['pixel_values_history']['dino'].shape)} OK")

    # Show what the special-token IDs decode to so we can eyeball the prompt structure.
    sample_action_ids = list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTIONS_CHUNK))
    sample_reason_ids = list(range(ACTION_REASON_TOKEN_BEGIN_IDX, ACTION_REASON_TOKEN_BEGIN_IDX + num_reason_token))
    print("-" * 60)
    print(f"  action token IDs {sample_action_ids} decode to: {repr(tokenizer.decode(sample_action_ids))}")
    print(f"  reason token IDs {sample_reason_ids} decode to: {repr(tokenizer.decode(sample_reason_ids))}")
    print("-" * 60)
    print("Decoded prompt (row 0, non-pad portion):")
    nonpad = batch['input_ids'][0][batch['attention_mask'][0].bool()]
    print(f"  {repr(tokenizer.decode(nonpad.tolist()))}")


if __name__ == "__main__":
    test_rlds_dataset()
    print("=" * 60)
    print("Dataset test passed.")
