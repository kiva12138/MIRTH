"""
OpenVLA-style PyTorch dataloader for local LeRobot datasets.

This module intentionally does not depend on pi0/pi05. It adapts LeRobot
frame datasets to the same per-sample contract produced by
``rlds_datasets.datasets.RLDSBatchTransform`` so the existing
``PaddedCollatorForActionPrediction`` can be reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config.config_vla import (
    ACTION_DIM,
    ACTION_REASON_TOKEN_BEGIN_IDX,
    PROPRIO_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    SINGLE_ACTION_CHUNK_TOKEN_INDEX,
    SINGLE_ACTION_TOKEN_INDEX,
)


DataDict = Dict[str, Any]


class LLaMA2PromptBuilder:
    def __init__(self) -> None:
        self.bos, self.eos = "<s>", "</s>"
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()
        if self.turn_count % 2 == 0:
            wrapped = f"In: {message}\nOut: "
        else:
            wrapped = f"{message if message != '' else ' '}{self.eos}"
        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()


def _import_lerobot_dataset_module():
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is required to read real LeRobot datasets. Install it in "
            "the active environment, then rerun this loader."
        ) from exc
    return lerobot_dataset


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, prefix=f"{key}."))
            out.update(_flatten(v, prefix=f"{key}/"))
        else:
            out[key] = v
    return out


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, Mapping):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.Tensor):
        return np.stack([x.detach().cpu().numpy() for x in obj], axis=0)
    return obj


def _as_float32(x: Any, dim: int | None = None, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if dim is not None:
        assert arr.ndim > 0, f"{name} must have at least one dimension, got shape {arr.shape}"
        assert arr.shape[-1] == dim, (
            f"{name} last dimension mismatch: expected {dim}, "
            f"got {arr.shape[-1]} with shape {arr.shape}"
        )
    return arr


def _image_from_array(x: Any) -> Image.Image:
    arr = np.asarray(x)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class LeRobotDataConfig:
    repo_id: str
    local_root: str | Path | None = None
    image_primary_key: str = "observation.images.main"
    image_wrist_keys: Sequence[str] = ("observation.images.front",)
    state_key: str = "observation.state"
    action_key: str = "action"
    task_index_key: str = "task_index"
    language_key: str | None = None
    default_prompt: str | None = None
    history_window_size: int = 20
    action_chunk_size: int = NUM_ACTIONS_CHUNK
    action_dim: int = ACTION_DIM
    proprio_dim: int = PROPRIO_DIM
    download_videos: bool = False
    video_backend: str | None = "pyav"

    def __post_init__(self) -> None:
        if self.history_window_size < 1:
            raise ValueError("history_window_size must be >= 1")
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be >= 1")


@dataclass
class LeRobotBatchTransform:
    base_tokenizer: Any
    image_transform: Any
    prompt_builder: Any | None = None
    action_token_type: str = "one_for_action_step"
    num_reason_tokens: int = 4
    use_reason_token: bool = True

    def __call__(self, sample: DataDict) -> DataDict:
        action_chunk = sample["action_chunk"]
        history_actions = sample["history_actions"]
        current_action = action_chunk[0]
        future_actions = action_chunk[1:]

        if self.action_token_type == "one_for_action_chunk":
            action_token_strings = self.base_tokenizer.decode(SINGLE_ACTION_CHUNK_TOKEN_INDEX)
        elif self.action_token_type == "one_for_action_step":
            action_token = self.base_tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
            action_token_strings = action_token * sample["action_chunk"].shape[0]
        elif self.action_token_type == "one_for_action_dim":
            action_token = self.base_tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
            action_token_strings = action_token * int(np.prod(sample["action_chunk"].shape))
        else:
            raise ValueError(f"Invalid action_token_type: {self.action_token_type}")

        reason_token_strings = ""
        if self.use_reason_token:
            reason_token_strings = self.base_tokenizer.decode(
                [ACTION_REASON_TOKEN_BEGIN_IDX + i for i in range(self.num_reason_tokens)]
            )

        language = str(sample["language_instruction"]).lower()
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language}?"},
            {
                "from": "gpt",
                "value": reason_token_strings + action_token_strings
                if self.use_reason_token
                else action_token_strings,
            },
        ]

        prompt_builder = self.prompt_builder.__class__() if self.prompt_builder is not None else LLaMA2PromptBuilder()
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        action_len = len(action_token_strings)
        reason_len = len(reason_token_strings)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[: -(action_len + reason_len + 1)] = IGNORE_INDEX
        labels[-1] = IGNORE_INDEX

        primary = self.image_transform(_image_from_array(sample["image_primary"]))
        primary_history = [self.image_transform(_image_from_array(x)) for x in sample["image_primary_history"]]

        wrist_keys = list(sample["wrist_images"].keys())
        wrist_current = {
            key: self.image_transform(_image_from_array(sample["wrist_images"][key]))
            for key in wrist_keys
        }
        wrist_history = {
            key: [self.image_transform(_image_from_array(x)) for x in sample["wrist_images_history"][key]]
            for key in wrist_keys
        }

        return {
            "input_ids": input_ids,
            "labels": labels,
            "current_action": current_action,
            "current_action_chunk": action_chunk,
            "future_actions": future_actions,
            "history_actions": history_actions,
            "pixel_values_primary": primary,
            "pixel_values_history": primary_history,
            "proprio": sample["proprio"],
            "proprio_history": sample["proprio_history"],
            "wrist_keys": wrist_keys,
            "wrist_pixel_values_dict": wrist_current,
            "wrist_pixel_values_history_dict": wrist_history,
            "pad_mask": sample["pad_mask"],
        }


class LeRobotOpenVLADataset(Dataset):
    def __init__(
        self,
        config: LeRobotDataConfig,
        batch_transform: LeRobotBatchTransform,
        lerobot_dataset: Dataset | None = None,
        tasks: Mapping[int, str] | None = None,
    ) -> None:
        self.config = config
        self.batch_transform = batch_transform

        if lerobot_dataset is None:
            lerobot_dataset, tasks = self._load_lerobot_dataset(config)
        self.dataset = lerobot_dataset
        self.tasks = {int(k): str(v) for k, v in dict(tasks or {}).items()}

        self._episode_bounds = self._build_episode_bounds()

    def _load_lerobot_dataset(self, config: LeRobotDataConfig) -> tuple[Dataset, Mapping[int, str]]:
        lerobot_dataset = _import_lerobot_dataset_module()
        root = Path(config.local_root) if config.local_root is not None else None
        if root is not None and not root.is_dir():
            raise FileNotFoundError(f"LeRobot local_root does not exist: {root}")

        meta = lerobot_dataset.LeRobotDatasetMetadata(config.repo_id, root=root)
        dataset = lerobot_dataset.LeRobotDataset(
            config.repo_id,
            root=root,
            download_videos=config.download_videos,
            video_backend=config.video_backend,
        )
        return dataset, _tasks_to_mapping(getattr(meta, "tasks", {}))

    def _build_episode_bounds(self) -> list[tuple[int, int]]:
        if not len(self.dataset):
            return []

        episode_data_index = getattr(self.dataset, "episode_data_index", None)
        if isinstance(episode_data_index, Mapping):
            from_values = episode_data_index.get("from")
            to_values = episode_data_index.get("to")
            if from_values is not None and to_values is not None:
                starts = np.asarray(_to_numpy(from_values), dtype=np.int64).reshape(-1)
                stops = np.asarray(_to_numpy(to_values), dtype=np.int64).reshape(-1)
                return [(int(s), int(t) - 1) for s, t in zip(starts, stops) if int(t) > int(s)]

        local_bounds = self._bounds_from_local_metadata()
        if local_bounds:
            return local_bounds

        bounds: list[tuple[int, int]] = []
        start = 0
        prev_episode = self._episode_index(0)
        for idx in range(1, len(self.dataset)):
            episode = self._episode_index(idx)
            if episode != prev_episode:
                bounds.append((start, idx - 1))
                start = idx
                prev_episode = episode
        bounds.append((start, len(self.dataset) - 1))
        return bounds

    def _bounds_from_local_metadata(self) -> list[tuple[int, int]]:
        if self.config.local_root is None:
            return []

        episodes_dir = Path(self.config.local_root) / "meta" / "episodes"
        if not episodes_dir.is_dir():
            return []

        try:
            import pandas as pd
        except ImportError:
            return []

        bounds: list[tuple[int, int]] = []
        for parquet_path in sorted(episodes_dir.glob("chunk-*/*.parquet")):
            df = pd.read_parquet(
                parquet_path,
                columns=["dataset_from_index", "dataset_to_index"],
            )
            for row in df.itertuples(index=False):
                start = int(row.dataset_from_index)
                stop = int(row.dataset_to_index)
                if stop > start:
                    bounds.append((start, stop - 1))
        return sorted(bounds)

    def _episode_index(self, idx: int) -> int:
        flat = _flatten(_to_numpy(self.dataset[idx]))
        value = flat.get("episode_index", flat.get("episode_index/0", 0))
        return int(np.asarray(value).item())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DataDict:
        start, end = self._bounds_for_index(idx)
        history_indices = [max(start, idx - offset) for offset in range(self.config.history_window_size - 1, 0, -1)]
        action_indices = [min(end, idx + offset) for offset in range(self.config.action_chunk_size)]

        current = self._raw(idx)
        history = [self._raw(i) for i in history_indices]
        future = [self._raw(i) for i in action_indices]

        sample = self._standardize(current, history, future, idx, start)
        return self.batch_transform(sample)

    def _bounds_for_index(self, idx: int) -> tuple[int, int]:
        for start, end in self._episode_bounds:
            if start <= idx <= end:
                return start, end
        raise IndexError(idx)

    def _raw(self, idx: int) -> dict[str, Any]:
        return _flatten(_to_numpy(self.dataset[idx]))

    def _get(self, raw: Mapping[str, Any], key: str) -> Any:
        if key in raw:
            return raw[key]
        slash_key = key.replace(".", "/")
        if slash_key in raw:
            return raw[slash_key]
        dot_key = key.replace("/", ".")
        if dot_key in raw:
            return raw[dot_key]
        raise KeyError(f"Missing key {key!r}. Available keys: {sorted(raw.keys())}")

    def _language(self, raw: Mapping[str, Any]) -> str:
        if self.config.language_key is not None:
            value = self._get(raw, self.config.language_key)
            if isinstance(value, bytes):
                return value.decode()
            return str(np.asarray(value).item() if hasattr(value, "item") else value)

        if self.config.task_index_key in raw or self.config.task_index_key.replace(".", "/") in raw:
            task_idx = int(np.asarray(self._get(raw, self.config.task_index_key)).item())
            if task_idx in self.tasks:
                return self.tasks[task_idx]

        if self.config.default_prompt is not None:
            return self.config.default_prompt
        raise ValueError("No language prompt found. Set language_key, tasks metadata, or default_prompt.")

    def _standardize(
        self,
        current: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        future: Sequence[Mapping[str, Any]],
        idx: int,
        episode_start: int,
    ) -> DataDict:
        image_primary_history = [self._get(raw, self.config.image_primary_key) for raw in history]
        wrist_history = {
            key: [self._get(raw, key) for raw in history]
            for key in self.config.image_wrist_keys
        }

        history_actions = np.stack(
            [
                _as_float32(
                    self._get(raw, self.config.action_key),
                    self.config.action_dim,
                    name=self.config.action_key,
                )
                for raw in history
            ],
            axis=0,
        )
        action_chunk = np.stack(
            [
                _as_float32(
                    self._get(raw, self.config.action_key),
                    self.config.action_dim,
                    name=self.config.action_key,
                )
                for raw in future
            ],
            axis=0,
        )

        proprio = _as_float32(
            self._get(current, self.config.state_key),
            self.config.proprio_dim,
            name=self.config.state_key,
        )
        proprio_history = np.stack(
            [
                _as_float32(
                    self._get(raw, self.config.state_key),
                    self.config.proprio_dim,
                    name=self.config.state_key,
                )
                for raw in history
            ],
            axis=0,
        )

        pad_mask = np.ones(self.config.history_window_size, dtype=bool)
        missing_history = max(0, self.config.history_window_size - 1 - (idx - episode_start))
        if missing_history:
            pad_mask[:missing_history] = False

        return {
            "language_instruction": self._language(current),
            "image_primary": self._get(current, self.config.image_primary_key),
            "image_primary_history": image_primary_history,
            "wrist_images": {
                key: self._get(current, key)
                for key in self.config.image_wrist_keys
            },
            "wrist_images_history": wrist_history,
            "proprio": proprio,
            "proprio_history": proprio_history,
            "action_chunk": action_chunk,
            "history_actions": history_actions,
            "pad_mask": pad_mask,
        }


def _tasks_to_mapping(tasks: Any) -> Mapping[int, str]:
    if hasattr(tasks, "iterrows"):
        return {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}
    if isinstance(tasks, Mapping):
        return {int(k): str(v) for k, v in tasks.items()}
    return {}


def create_lerobot_dataloader(
    data_config: LeRobotDataConfig,
    batch_transform: LeRobotBatchTransform,
    collator,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    dataset = LeRobotOpenVLADataset(data_config, batch_transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def validate_lerobot_batch(batch: Mapping[str, Any], batch_size: int, history_window_size: int) -> None:
    history = history_window_size - 1
    required = {
        "input_ids",
        "attention_mask",
        "labels",
        "current_action",
        "current_action_chunk",
        "history_actions",
        "pixel_values",
        "pixel_values_history",
        "proprio",
        "proprio_history",
        "pad_mask",
    }
    missing = sorted(required - set(batch.keys()))
    if missing:
        raise AssertionError(f"Missing collated batch keys: {missing}")

    if batch["current_action"].shape[0] != batch_size:
        raise AssertionError(f"batch size mismatch: {batch['current_action'].shape[0]} != {batch_size}")
    if batch["history_actions"].shape[1] != history:
        raise AssertionError(f"history length mismatch: {batch['history_actions'].shape[1]} != {history}")
    if batch["pad_mask"].shape != (batch_size, history_window_size):
        raise AssertionError(f"pad_mask shape mismatch: {tuple(batch['pad_mask'].shape)}")
    for encoder_name in ("dino", "siglip"):
        if encoder_name not in batch["pixel_values"]:
            raise AssertionError(f"pixel_values missing encoder {encoder_name!r}")
        if encoder_name not in batch["pixel_values_history"]:
            raise AssertionError(f"pixel_values_history missing encoder {encoder_name!r}")
        if batch["pixel_values_history"][encoder_name].shape[2] != history:
            raise AssertionError(
                f"{encoder_name} image history length mismatch: "
                f"{batch['pixel_values_history'][encoder_name].shape[2]} != {history}"
            )
