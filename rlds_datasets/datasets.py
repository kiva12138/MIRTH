"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from models.llm_llama2 import LLaMA2PurePromptBuilder as PromptBuilder
from models.vision_encoders import ImageTransform
from rlds_datasets.rlds import make_interleaved_dataset
from rlds_datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from utils.overwatch import initialize_overwatch
from config.config_vla import IGNORE_INDEX, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK, SINGLE_ACTION_TOKEN_INDEX, SINGLE_ACTION_CHUNK_TOKEN_INDEX, ACTION_REASON_TOKEN_BEGIN_IDX, ACTION_DIM

overwatch = initialize_overwatch(__name__)
overwatch.info("Datamix could be found in datasets.rlds.oxe.miextures")
overwatch.info("load_camera_views could be found in datasets.rlds.oxe.configs")


@dataclass
class RLDSBatchTransform:
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder: PromptBuilder = None
    action_token_type: str = "one_for_action_step" # options: "one_for_action_chunk", "one_for_action_step", "one_for_action_dim"
    num_reason_tokens: int = 4
    use_reason_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        window_size = rlds_batch["observation"]["image_primary"].shape[0]

        # Process action chunk
        current_action_chunk = rlds_batch["action"][window_size-1:]
        history_actions = rlds_batch["action"][:window_size-1]
        current_action, future_actions = current_action_chunk[0], current_action_chunk[1:]
        
        # Build prompt
        if self.action_token_type == "one_for_action_chunk":
            action_token_strings = self.base_tokenizer.decode(SINGLE_ACTION_CHUNK_TOKEN_INDEX)
        elif self.action_token_type == "one_for_action_step":
            action_token_string = self.base_tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
            action_token_strings = action_token_string * NUM_ACTIONS_CHUNK
        elif self.action_token_type == "one_for_action_dim":
            action_token_string = self.base_tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
            action_token_strings = action_token_string * (NUM_ACTIONS_CHUNK * ACTION_DIM)
        else:
            raise ValueError(f"Invalid action_token_type: {self.action_token_type}")
        
        action_chunk_len = len(action_token_strings)
        if self.use_reason_token:
            reason_token_strings = self.base_tokenizer.decode([ACTION_REASON_TOKEN_BEGIN_IDX + i for i in range(self.num_reason_tokens)])
            reason_token_len = len(reason_token_strings)
        else:
            reason_token_len = 0
        language = rlds_batch["task"]["language_instruction"].decode().lower()
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language}?"},
            {"from": "gpt", "value": reason_token_strings + action_token_strings if self.use_reason_token else action_token_strings},
        ]
        prompt_builder = self.prompt_builder.__class__()
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(action_chunk_len + reason_token_len + 1)] = IGNORE_INDEX
        labels[-1] = IGNORE_INDEX # Here we ignore the EOS token

        # Process images
        image_primary = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        image_primary_history = [Image.fromarray(rlds_batch["observation"]["image_primary"][i])  for i in range(window_size - 1)]
        pixel_values_primary = self.image_transform(image_primary)
        pixel_values_history = [self.image_transform(image_) for image_ in image_primary_history]
        del image_primary, image_primary_history
        
        # Process proprioception
        if "proprio" in rlds_batch["observation"].keys():
            proprio = rlds_batch["observation"]["proprio"][-1]
            history_proprio = rlds_batch["observation"]["proprio"][:-1]
        else:
            proprio = None
            history_proprio = None
            
        # Process wrist images
        wrist_keys = [k for k in rlds_batch["observation"].keys() if "wrist" in k]
        wrist_pixel_values_dict = {key: None for key in wrist_keys}
        wrist_pixel_values_history_dict = {key: [] for key in wrist_keys}
        for key in wrist_keys:
            img_wrist = Image.fromarray(rlds_batch["observation"][key][-1])
            wrist_pixel_values_dict[key] = self.image_transform(img_wrist)
            
            history_wrist_imgs = [Image.fromarray(rlds_batch["observation"][key][i]) for i in range(window_size - 1)]
            history_wrist_pixels = [self.image_transform(img) for img in history_wrist_imgs]
            wrist_pixel_values_history_dict[key] = history_wrist_pixels

        # Pad mask for variable-length sequences
        pad_mask = rlds_batch["observation"]['pad_mask']

        return_dict = dict(
                        input_ids=input_ids, 
                        labels=labels, 
                        current_action=current_action,
                        current_action_chunk=current_action_chunk,
                        future_actions=future_actions,
                        history_actions=history_actions,
                        pixel_values_primary=pixel_values_primary, 
                        pixel_values_history=pixel_values_history, 
                        proprio=proprio,
                        proprio_history=history_proprio,
                        wrist_keys=wrist_keys,
                        wrist_pixel_values_dict=wrist_pixel_values_dict,
                        wrist_pixel_values_history_dict=wrist_pixel_values_history_dict,
                        pad_mask=pad_mask,
                    )        

        return return_dict
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        load_camera_views: Tuple[str, str] = ("primary", "wrist"),
        load_proprio: bool = True,
        train: bool = True,
        image_aug: bool = False,
        history_window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.history_window_size = history_window_size
        # self._batch_count = 0  # Track batches for periodic cleanup

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=history_window_size,                    # History window size for temporal context
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        import gc
        import tensorflow as tf

        # Strategy: Recreate iterator periodically to prevent TensorFlow cache buildup
        chunk_size = 1000  # Recreate iterator every 1000 samples

        while True:  # Outer infinite loop for repeated iteration
            # Create a fresh iterator with limited lifetime
            iterator = self.dataset.as_numpy_iterator()

            # Process exactly chunk_size samples with this iterator
            for i in range(chunk_size):
                rlds_batch = next(iterator)
                yield self.batch_transform(rlds_batch)

            # After processing chunk_size samples, clean up this iterator
            del iterator
            gc.collect()
            tf.keras.backend.clear_session()
            print("Cleared TensorFlow session to prevent cache buildup.")

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

