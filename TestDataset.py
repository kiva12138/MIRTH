import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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


DATA_ROOT_DIR = r"E:\LeRobotKitchenNewRLDS"
DATASET_NAME = "lerobot_kitchen_new_basic_tasks"
HF_TOKEN = "your_hf_token"  # replace with your Hugging Face token if needed
OUTPUT_DIR = Path("./outputs")
VIS_MAX_SAMPLES = 32
VIS_PROGRESS_EVERY = 10


def image_to_pil(value) -> Image.Image:
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")


def vector_text(value, precision: int = 3, max_items: int = 12) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    shown = arr[:max_items].tolist()
    suffix = ", ..." if arr.size > max_items else ""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in shown) + suffix + "]"


def scalar_text(value) -> str:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return ""
    item = arr[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def save_rlds_visualization(rlds_batch, idx: int, output_root: Path) -> Path:
    observation = rlds_batch["observation"]
    task = rlds_batch["task"]
    primary = image_to_pil(observation["image_primary"][-1])
    wrist = image_to_pil(observation["image_wrist"][-1])
    current_action_index = int(np.asarray(observation["image_primary"]).shape[0] - 1)
    current_action_chunk = rlds_batch["action"][current_action_index:]

    label_height = 24
    text_lines = [
        f"sample={idx} dataset={scalar_text(rlds_batch['dataset_name'])} timestep={scalar_text(observation['timestep'][-1])}",
        f"task: {scalar_text(task['language_instruction'])}",
        f"current_action: {vector_text(rlds_batch['action'][current_action_index])}",
        f"current_action_chunk_shape: {tuple(current_action_chunk.shape)} full_action_shape: {tuple(rlds_batch['action'].shape)}",
        f"proprio: {vector_text(observation['proprio'][-1])}",
        f"proprio_shape: {tuple(observation['proprio'].shape)} pad_mask: {np.asarray(observation['pad_mask']).astype(int).tolist()}",
    ]
    text_height = 18 * len(text_lines) + 12
    image_height = max(primary.height, wrist.height)
    width = primary.width + wrist.width

    canvas = Image.new("RGB", (width, label_height + image_height + text_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for name, image in (("primary", primary), ("wrist", wrist)):
        draw.rectangle((x, 0, x + image.width, label_height), fill=(30, 30, 30))
        draw.text((x + 8, 5), name, fill=(255, 255, 255))
        canvas.paste(image, (x, label_height))
        x += image.width

    y = label_height + image_height + 6
    for line in text_lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18

    out_dir = output_root / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rlds_sample_{idx:06d}.jpg"
    canvas.save(out_path, quality=92)
    return out_path


def visualize_rlds_dataset(dataset: RLDSDataset, output_root: Path, max_samples: int) -> None:
    print("-" * 60)
    print(f"Saving {max_samples} RLDS visualizations to {output_root / DATASET_NAME}")
    print("-" * 60)

    started_at = time.time()
    iterator = dataset.dataset.as_numpy_iterator()
    try:
        for idx in range(max_samples):
            save_rlds_visualization(next(iterator), idx, output_root)
            if VIS_PROGRESS_EVERY > 0 and ((idx + 1) % VIS_PROGRESS_EVERY == 0 or idx + 1 == max_samples):
                elapsed = time.time() - started_at
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                print(f"visualized {idx + 1}/{max_samples}, rate={rate:.2f} samples/s", flush=True)
    finally:
        del iterator


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
    visualize_rlds_dataset(dataset, OUTPUT_DIR, VIS_MAX_SAMPLES)

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
