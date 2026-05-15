from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Callable

import torch
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from config.config_vla import IGNORE_INDEX
import psutil
import os


def get_memory_usage():
    """Get current CPU and GPU memory usage in MB."""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / 1024 / 1024  # RSS in MB

    # GPU memory
    if torch.cuda.is_available():
        gpu_mem_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
    else:
        gpu_mem_allocated_mb = 0.0
        gpu_mem_reserved_mb = 0.0

    return {
        'cpu_memory_mb': cpu_mem_mb,
        'gpu_memory_allocated_mb': gpu_mem_allocated_mb,
        'gpu_memory_reserved_mb': gpu_mem_reserved_mb,
    }


def as_float(v, default=0.0):
    if v is None:
        return float(default)
    if isinstance(v, torch.Tensor):
        # Detach and move to CPU to break graph and device ties
        return float(v.detach().cpu().item())
    return float(v)

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()}

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # instance keys:
        # ['input_ids', 'labels', 
        #  'current_action', 'current_action_chunk', 'future_actions', 'history_actions', 
        #  'pixel_values_primary', 'pixel_values_history', 
        #  'proprio', 'proprio_history', # Could be None
        #  'wrist_keys', 'wrist_pixel_values_dict', 'wrist_pixel_values_history_dict']

        # Process text inputs
        # For now, we only support Tokenizers with `padding_side = "right"` during training => Handle padding via RNN Utils => `pad_sequence`
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)
        
        # Stack all actions
        current_action = [torch.from_numpy(np.copy(instance["current_action"])) for instance in instances]
        current_action = torch.stack(current_action, dim=0)
        current_action_chunk = [torch.from_numpy(np.copy(instance["current_action_chunk"])) for instance in instances]
        current_action_chunk = torch.stack(current_action_chunk, dim=0)
        history_actions = [torch.from_numpy(np.copy(instance["history_actions"])) for instance in instances]
        history_actions = torch.stack(history_actions, dim=0)

        # Process primary image inputs
        pixel_values_return = {}
        pixel_values_primary = [instance["pixel_values_primary"] for instance in instances]
        assert isinstance(pixel_values_primary[0], dict), f"Unsupported `pixel_values` type = {type(pixel_values_primary[0])}"
        pixel_values_return['dino'] = torch.stack([instance['dino'] for instance in pixel_values_primary], dim=0).unsqueeze(1)     # [B, 1, C, H, W]
        pixel_values_return['siglip'] = torch.stack([instance['siglip'] for instance in pixel_values_primary], dim=0).unsqueeze(1) # [B, 1, C, H, W]
        # Process wrist image inputs
        for wrist_key in instances[0]['wrist_keys']:
            current_wrist_pixel_values = [instance['wrist_pixel_values_dict'][wrist_key] for instance in instances]
            current_wrist_pixel_values_dino = torch.stack([instance['dino'] for instance in current_wrist_pixel_values], dim=0)
            current_wrist_pixel_values_siglip = torch.stack([instance['siglip'] for instance in current_wrist_pixel_values], dim=0)
            pixel_values_return['dino'] = torch.cat([pixel_values_return['dino'], current_wrist_pixel_values_dino.unsqueeze(1)], dim=1)       # [B, num_cameras, C, H, W]
            pixel_values_return['siglip'] = torch.cat([pixel_values_return['siglip'], current_wrist_pixel_values_siglip.unsqueeze(1)], dim=1) # [B, num_cameras, C, H, W]

        # Process primary image history inputs
        pixel_values_history_return = {}
        pixel_values_primary_history = [instance["pixel_values_history"] for instance in instances]
        pixel_values_history_return['dino'] = torch.stack([torch.stack([pv['dino'] for pv in instance], dim=0) for instance in pixel_values_primary_history], dim=0).unsqueeze(1)     # [B, 1, His, C, H, W]
        pixel_values_history_return['siglip'] = torch.stack([torch.stack([pv['siglip'] for pv in instance], dim=0) for instance in pixel_values_primary_history], dim=0).unsqueeze(1) # [B, 1, His, C, H, W]
        # Process wrist image history inputs
        for wrist_key in instances[0]['wrist_keys']:
            current_wrist_pixel_values_history = [instance['wrist_pixel_values_history_dict'][wrist_key] for instance in instances]
            current_wrist_pixel_values_history_dino = torch.stack([torch.stack([pv['dino'] for pv in instance], dim=0) for instance in current_wrist_pixel_values_history], dim=0) # [B, His, C, H, W]
            current_wrist_pixel_values_history_siglip = torch.stack([torch.stack([pv['siglip'] for pv in instance], dim=0) for instance in current_wrist_pixel_values_history], dim=0) # [B, His, C, H, W]
            pixel_values_history_return['dino'] = torch.cat([pixel_values_history_return['dino'], current_wrist_pixel_values_history_dino.unsqueeze(1)], dim=1)       # [B, num_cameras, His, C, H, W]
            pixel_values_history_return['siglip'] = torch.cat([pixel_values_history_return['siglip'], current_wrist_pixel_values_history_siglip.unsqueeze(1)], dim=1) # [B, num_cameras, His, C, H, W]

        # Process proprio
        if instances[0]["proprio"] is not None:
            proprio = torch.stack([torch.from_numpy(instance["proprio"].copy()) for instance in instances], dim=0)  # [B, Proprio_dim]
            proprio_history = [torch.from_numpy(instance["proprio_history"].copy()) for instance in instances]
            proprio_history = torch.stack(proprio_history, dim=0) # [B, His, Proprio_dim]
        else:
            proprio = None
            proprio_history = None

        # Process pad_mask
        pad_mask = torch.stack([torch.from_numpy(np.copy(instance["pad_mask"])) for instance in instances], dim=0)

        output = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            current_action=current_action,
            current_action_chunk=current_action_chunk,
            history_actions=history_actions,
            pixel_values= pixel_values_return,
            pixel_values_history=pixel_values_history_return,
            proprio=proprio,
            proprio_history=proprio_history,
            pad_mask=pad_mask,
        )
            
        return output
    