import argparse
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


def find_first_lerobot_dataset(root: Path) -> Path:
    if (root / "data").is_dir() and (root / "meta").is_dir():
        return root
    for path in root.rglob("meta"):
        candidate = path.parent
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"No LeRobot dataset folder found under {root}")


def slugify(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).lower()).strip("_")
    return re.sub(r"_+", "_", slug) or "dataset"


def to_numpy(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(value, Mapping):
        return {k: to_numpy(v) for k, v in value.items()}
    return value


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{full_key}."))
        else:
            out[full_key] = value
    return out


def get_value(flat: Mapping[str, Any], key: str) -> Any:
    if key in flat:
        return flat[key]
    slash_key = key.replace(".", "/")
    if slash_key in flat:
        return flat[slash_key]
    dot_key = key.replace("/", ".")
    if dot_key in flat:
        return flat[dot_key]
    raise KeyError(f"Missing key {key!r}. Available keys: {sorted(flat)}")


def scalar(value: Any, default: int = -1) -> int:
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def image_to_pil(value: Any) -> Image.Image:
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


def vector_text(value: Any, precision: int = 3) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr.tolist()) + "]"


def task_text(flat: Mapping[str, Any], default_prompt: str) -> str:
    task = flat.get("task")
    if task is not None:
        if isinstance(task, bytes):
            return task.decode()
        return str(np.asarray(task).item() if hasattr(task, "item") else task)
    return default_prompt


def resize_for_canvas(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    height = int(image.height * (max_width / image.width))
    return image.resize((max_width, height), Image.BILINEAR)


def save_visualization(
    flat: Mapping[str, Any],
    idx: int,
    output_root: Path,
    *,
    dataset_name: str,
    default_prompt: str,
    primary_image_key: str,
    wrist_image_key: str,
    state_key: str,
    action_key: str,
    max_image_width: int,
    jpeg_quality: int,
) -> Path:
    episode_idx = scalar(flat.get("episode_index", -1))
    frame_idx = scalar(flat.get("frame_index", idx))
    timestamp = np.asarray(flat.get("timestamp", 0.0)).reshape(-1)[0]

    images = [
        ("main", resize_for_canvas(image_to_pil(get_value(flat, primary_image_key)), max_image_width)),
        ("front", resize_for_canvas(image_to_pil(get_value(flat, wrist_image_key)), max_image_width)),
    ]

    label_height = 24
    text_lines = [
        f"idx={idx} episode={episode_idx} frame={frame_idx} timestamp={float(timestamp):.3f}",
        f"task: {task_text(flat, default_prompt)}",
        f"action: {vector_text(get_value(flat, action_key))}",
        f"state:  {vector_text(get_value(flat, state_key))}",
    ]
    text_height = 18 * len(text_lines) + 12
    image_height = max(image.height for _, image in images)
    width = sum(image.width for _, image in images)

    canvas = Image.new("RGB", (width, label_height + image_height + text_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for name, image in images:
        draw.rectangle((x, 0, x + image.width, label_height), fill=(30, 30, 30))
        draw.text((x + 8, 5), name, fill=(255, 255, 255))
        canvas.paste(image, (x, label_height))
        x += image.width

    y = label_height + image_height + 6
    for line in text_lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18

    episode_dir = output_root / dataset_name / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    out_path = episode_dir / f"frame_{frame_idx:06d}_idx_{idx:08d}.jpg"
    canvas.save(out_path, quality=jpeg_quality)
    return out_path


def build_lerobot_dataset(args):
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    return lerobot_dataset.LeRobotDataset(
        args.repo_id,
        root=args.local_root,
        download_videos=False,
        video_backend=args.video_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, default=Path(r"E:\LeRobotKitchenNew"))
    parser.add_argument("--repo-id", default="local_lerobot")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--primary-image-key", default="observation.images.main")
    parser.add_argument("--wrist-image-key", default="observation.images.front")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "video_reader", "torchcodec"))
    parser.add_argument("--max-image-width", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    args.local_root = find_first_lerobot_dataset(args.local_root)
    dataset_name = slugify(args.local_root.name)
    dataset = build_lerobot_dataset(args)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    print(f"Visualizing {total}/{len(dataset)} samples from {args.local_root}", flush=True)
    print(f"Output directory: {args.output_dir / dataset_name}", flush=True)

    started_at = time.time()
    for idx in range(total):
        flat = flatten(to_numpy(dataset[idx]))
        save_visualization(
            flat,
            idx,
            args.output_dir,
            dataset_name=dataset_name,
            default_prompt=args.local_root.name,
            primary_image_key=args.primary_image_key,
            wrist_image_key=args.wrist_image_key,
            state_key=args.state_key,
            action_key=args.action_key,
            max_image_width=args.max_image_width,
            jpeg_quality=args.jpeg_quality,
        )

        if args.progress_every > 0 and ((idx + 1) % args.progress_every == 0 or idx + 1 == total):
            elapsed = time.time() - started_at
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            print(f"visualized {idx + 1}/{total}, rate={rate:.2f} samples/s", flush=True)

    print("Visualization complete.", flush=True)


if __name__ == "__main__":
    main()
