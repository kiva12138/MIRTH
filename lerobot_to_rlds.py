"""
Convert local LeRobot datasets to TFDS/RLDS datasets.

The output is compatible with the existing RLDS loader path:
    tfds.builder(name, data_dir=<output_root>)
    dlimp.DLataset.from_rlds(...)

By default the script scans a LeRobot root that contains many task folders and
writes one RLDS dataset per task. Dataset names are generated from folder names,
for example:
    "Place the banana in the plate on the right"
        -> "place_the_banana_in_the_plate_on_the_right"
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image


DEFAULT_LEROBOT_ROOT = Path(r"E:\DATA_TRIMMED")
DEFAULT_RLDS_ROOT = Path(r"E:\RLDS_DATA")


@dataclass(frozen=True)
class LeRobotSource:
    name: str
    path: Path


def slugify(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def builder_class_name(dataset_name: str) -> str:
    parts = re.findall(r"[0-9a-zA-Z]+", dataset_name)
    stem = "".join(part[:1].upper() + part[1:] for part in parts)
    if not stem or not stem[0].isalpha():
        stem = f"Dataset{stem}"
    return f"{stem}RldsBuilder"


def find_lerobot_datasets(root: Path) -> list[LeRobotSource]:
    if (root / "data").is_dir() and (root / "meta").is_dir():
        return [LeRobotSource(slugify(root.name), root)]

    found: list[LeRobotSource] = []
    for meta_dir in sorted(root.rglob("meta")):
        dataset_dir = meta_dir.parent
        if (dataset_dir / "data").is_dir():
            found.append(LeRobotSource(slugify(dataset_dir.name), dataset_dir))

    # Preserve first occurrence if duplicate task names exist under different categories.
    unique: dict[str, LeRobotSource] = {}
    for source in found:
        if source.name in unique:
            raise ValueError(
                f"Duplicate dataset name {source.name!r}: {unique[source.name].path} and {source.path}. "
                "Use --single-dataset on one folder or rename one output dataset."
            )
        unique[source.name] = source
    return list(unique.values())


def import_lerobot():
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
    except ImportError as exc:
        raise ImportError("This converter requires the `lerobot` package in the active environment.") from exc
    return lerobot_dataset


def import_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError("This converter requires `tensorflow-datasets` in the active environment.") from exc
    return tfds


def to_numpy(obj: Any) -> Any:
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(obj, Mapping):
        return {k: to_numpy(v) for k, v in obj.items()}
    return obj


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
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
    raise KeyError(f"Missing key {key!r}. Available keys: {sorted(flat)}")


def as_uint8_hwc(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        raise ValueError(f"Expected image rank 3, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def encode_jpeg(image: Any, quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(as_uint8_hwc(image)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def vector(value: Any, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if dim is not None:
        if arr.shape[0] > dim:
            arr = arr[:dim]
        elif arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]), mode="constant")
    return arr.astype(np.float32)


def read_info(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tasks(dataset_dir: Path) -> dict[int, str]:
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}

    import pandas as pd

    tasks = pd.read_parquet(tasks_path)
    if "task_index" in tasks.columns:
        text_col = "task" if "task" in tasks.columns else tasks.columns[0]
        return {int(row["task_index"]): str(row[text_col]) for _, row in tasks.iterrows()}
    if len(tasks.columns) >= 2:
        return {int(row[tasks.columns[0]]): str(row[tasks.columns[1]]) for _, row in tasks.iterrows()}
    return {int(i): str(v) for i, v in enumerate(tasks.iloc[:, 0].tolist())}


def episode_bounds(dataset: Any, dataset_dir: Path) -> list[tuple[int, int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if isinstance(episode_data_index, Mapping):
        starts = episode_data_index.get("from")
        stops = episode_data_index.get("to")
        if starts is not None and stops is not None:
            starts = np.asarray(to_numpy(starts), dtype=np.int64).reshape(-1)
            stops = np.asarray(to_numpy(stops), dtype=np.int64).reshape(-1)
            return [(int(s), int(t) - 1) for s, t in zip(starts, stops) if int(t) > int(s)]

    import pandas as pd

    bounds: list[tuple[int, int]] = []
    for path in sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path, columns=["dataset_from_index", "dataset_to_index"])
        bounds.extend((int(row.dataset_from_index), int(row.dataset_to_index) - 1) for row in df.itertuples())
    if not bounds:
        raise ValueError(f"No episode bounds found for {dataset_dir}")
    return sorted(bounds)


def make_builder_class(
    args,
    source: LeRobotSource,
    action_dim: int,
    state_dim: int,
    primary_image_shape: tuple[int, int, int],
    wrist_image_shape: tuple[int, int, int],
):
    tfds = import_tfds()

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"RLDS conversion of {source.path}",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=primary_image_shape,
                                        encoding_format="jpeg",
                                    ),
                                    "front_image": tfds.features.Image(
                                        shape=wrist_image_shape,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(state_dim,),
                                        dtype=np.float32,
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(action_dim,),
                                dtype=np.float32,
                            ),
                            "language_instruction": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "reward": np.float32,
                            "discount": np.float32,
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "episode_index": np.int64,
                            "source_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        yield from generate_examples(args, source)

    return type(
        builder_class_name(source.name),
        (tfds.core.GeneratorBasedBuilder,),
        {
            "__module__": __name__,
            "VERSION": tfds.core.Version(args.version),
            "RELEASE_NOTES": {args.version: "Converted from local LeRobot dataset."},
            "_info": _info,
            "_split_generators": _split_generators,
            "_generate_examples": _generate_examples,
        },
    )


def generate_examples(args, source: LeRobotSource):
    lerobot_dataset = import_lerobot()

    dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        root=source.path,
        download_videos=False,
        video_backend=args.video_backend,
    )
    bounds = episode_bounds(dataset, source.path)
    tasks = read_tasks(source.path)

    for episode_number, (start, end) in enumerate(bounds):
        steps = []
        for idx in range(start, end + 1):
            raw = flatten(to_numpy(dataset[idx]))
            task_index = int(np.asarray(get_value(raw, args.task_index_key)).item())
            language = tasks.get(task_index, args.default_prompt or source.path.name)
            is_last = idx == end

            steps.append(
                {
                    "observation": {
                        "image": encode_jpeg(get_value(raw, args.primary_image_key), args.jpeg_quality),
                        "front_image": encode_jpeg(get_value(raw, args.wrist_image_key), args.jpeg_quality),
                        "state": vector(get_value(raw, args.state_key), args.state_dim),
                    },
                    "action": vector(get_value(raw, args.action_key), args.action_dim),
                    "language_instruction": language,
                    "is_first": idx == start,
                    "is_last": is_last,
                    "is_terminal": is_last,
                    "reward": np.float32(1.0 if is_last else 0.0),
                    "discount": np.float32(0.0 if is_last else 1.0),
                }
            )

        yield str(episode_number), {
            "steps": steps,
            "episode_metadata": {
                "episode_index": np.int64(episode_number),
                "source_path": str(source.path),
            },
        }


def infer_shapes(source: LeRobotSource, args) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int]]:
    info = read_info(source.path)
    features = info.get("features", {})
    action_shape = features.get(args.action_key, {}).get("shape", [args.action_dim])
    state_shape = features.get(args.state_key, {}).get("shape", [args.state_dim])
    primary_image_shape = tuple(features.get(args.primary_image_key, {}).get("shape", [480, 640, 3]))
    wrist_image_shape = tuple(features.get(args.wrist_image_key, {}).get("shape", [480, 640, 3]))
    action_dim = int(args.action_dim or action_shape[-1])
    state_dim = int(args.state_dim or state_shape[-1])
    return action_dim, state_dim, primary_image_shape, wrist_image_shape


def convert_one(source: LeRobotSource, args) -> None:
    tfds = import_tfds()
    action_dim, state_dim, primary_image_shape, wrist_image_shape = infer_shapes(source, args)
    builder_cls = make_builder_class(
        args,
        source,
        action_dim,
        state_dim,
        primary_image_shape,
        wrist_image_shape,
    )

    dataset_root = args.output_root / source.name
    if args.overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    builder = builder_cls(data_dir=str(args.output_root))
    print(
        f"Converting {source.path} -> {builder.data_dir} "
        f"(action_dim={action_dim}, state_dim={state_dim})",
        flush=True,
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            manual_dir=str(source.path),
            download_mode=(
                tfds.download.GenerateMode.FORCE_REDOWNLOAD
                if args.overwrite
                else tfds.download.GenerateMode.REUSE_DATASET_IF_EXISTS
            ),
        )
    )
    print(f"Done: {source.name}", flush=True)


def dry_run(sources: Iterable[LeRobotSource], args) -> None:
    for source in sources:
        info = read_info(source.path)
        action_dim, state_dim, primary_image_shape, wrist_image_shape = infer_shapes(source, args)
        print(
            f"{source.name}: path={source.path}, "
            f"episodes={info.get('total_episodes')}, frames={info.get('total_frames')}, "
            f"action_dim={action_dim}, state_dim={state_dim}, "
            f"primary_image_shape={primary_image_shape}, wrist_image_shape={wrist_image_shape}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--single-dataset", type=Path, default=None)
    parser.add_argument("--repo-id", default="local_lerobot")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--primary-image-key", default="observation.images.main")
    parser.add_argument("--wrist-image-key", default="observation.images.front")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--task-index-key", default="task_index")
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "video_reader", "torchcodec"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.single_dataset if args.single_dataset is not None else args.lerobot_root
    sources = find_lerobot_datasets(root)
    if not sources:
        raise FileNotFoundError(f"No LeRobot datasets found under {root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(sources)} LeRobot dataset(s).", flush=True)

    if args.dry_run:
        dry_run(sources, args)
        return

    for source in sources:
        convert_one(source, args)


if __name__ == "__main__":
    main()
