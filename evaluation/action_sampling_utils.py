from utils.ignore_warning import ignore_warnings
ignore_warnings()

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import random
import torch
import tqdm
import wandb
from PIL import Image
import tensorflow as tf
import math

from config.constants_new import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, NUM_ACTIONS_CHUNK
from config.constants_new import SINGLE_ACTION_TOKEN_INDEX, SINGLE_ACTION_CHUNK_TOKEN_INDEX, ACTION_REASON_TOKEN_BEGIN_IDX, ACTION_DIM, IGNORE_INDEX, PROPRIO_DIM
from VLAMiniCodes.models.vla_model import OpenVLAOFTConfig, OpenVLAOFTForVision2Seq

OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def process_action(action):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action = invert_gripper_action(action)

    return action


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    # For Libero we delete zero column
    if proprio_high.size == 9 and proprio_low.size == 9 and proprio.size == 8:
        proprio = np.concatenate([proprio[:6], [0], proprio[6:]])

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def get_vla_action(
    cfg,
    model: OpenVLAOFTForVision2Seq,
    obs: Dict[str, Any],
    task_label: str,
    prompt_builder,
    action_history_tensor,
    pixel_values_history_tensor,
    proprio_history_tensor,
    pad_mask,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        model: The VLA model
        obs: Observation dictionary
        task_label: Text description of the task
    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
        assert len(all_images) == cfg.num_images_in_input, (f"Expected {cfg.num_images_in_input} images but got {len(all_images)}!")

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)
        all_images = [model.image_transform(image) for image in all_images]
        all_pixel_values = {
            'dino': torch.stack([v['dino'].unsqueeze(0) for v in all_images], dim=1),       # [B, num_cameras, C, H, W]
            'siglip': torch.stack([v['siglip'].unsqueeze(0) for v in all_images], dim=1),   # [B, num_cameras, C, H, W]
        }

        # Build VLA prompt
        if cfg.use_original_action_tokens:
            action_token_string = model.tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
            action_token_strings = action_token_string * NUM_ACTIONS_CHUNK * ACTION_DIM
        else:
            if cfg.one_token_for_action_chunk:
                action_token_strings = model.tokenizer.decode(SINGLE_ACTION_CHUNK_TOKEN_INDEX)
            else:
                action_token_string = model.tokenizer.decode(SINGLE_ACTION_TOKEN_INDEX)
                action_token_strings = action_token_string * NUM_ACTIONS_CHUNK
        action_chunk_len = len(action_token_strings)
        if cfg.use_reason_token:
            reason_token_strings = model.tokenizer.decode([ACTION_REASON_TOKEN_BEGIN_IDX + i for i in range(cfg.num_reason_token)])
            reason_token_len = len(reason_token_strings)
        else:
            reason_token_len = 0
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {task_label.lower()}?"},
            {"from": "gpt", "value": reason_token_strings + action_token_strings if cfg.use_reason_token else action_token_strings},
        ]
        prompt_builder = prompt_builder.__class__()
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = model.tokenizer([prompt_builder.get_prompt()], add_special_tokens=True, return_tensors="pt").input_ids
        labels = input_ids.clone()
        labels[:, : -(action_chunk_len + reason_token_len + 1)] = IGNORE_INDEX
        labels[:, -1] = IGNORE_INDEX # Here we ignore the EOS token
        attention_mask = torch.ones_like(input_ids)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio_norm_stats = model.norm_stats[cfg.unnorm_keys]["proprio"]
            proprio = normalize_proprio(obs["state"], proprio_norm_stats)
            proprio = torch.Tensor(proprio)


        dummy_current_action = torch.zeros(size=(1, ACTION_DIM))
        dummy_current_action_chunk = torch.zeros(size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM))
        action, normalized_actions = model.predict_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            current_action=dummy_current_action,
            current_action_chunk=dummy_current_action_chunk,
            history_actions=action_history_tensor,
            pixel_values=all_pixel_values,
            pixel_values_history=pixel_values_history_tensor,
            proprio=proprio,
            proprio_history=proprio_history_tensor,
            pad_mask=pad_mask,
            unnorm_key=cfg.unnorm_keys,       
        )

    # Return action chunk as list of actions
    return [action[i] for i in range(len(action))], normalized_actions, all_pixel_values, proprio








