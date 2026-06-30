"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any, Dict

import tensorflow as tf

from rlds_datasets.rlds.oxe.configs import LEROBOT_KITCHEN_NEW_TASKS
from rlds_datasets.rlds.oxe.utils.droid_utils import droid_baseact_transform, droid_finetuning_transform
from rlds_datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory

def lerobot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    tf.debugging.assert_equal(
        tf.shape(trajectory["action"])[-1],
        6,
        message="LeRobotKitchenNew action must be 6D.",
    )

    trajectory["observation"]["EEF_state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    tf.debugging.assert_equal(
        tf.shape(trajectory["observation"]["EEF_state"])[-1],
        6,
        message="LeRobotKitchenNew proprio/state must be 6D.",
    )
    return trajectory



# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,

}

OXE_STANDARDIZATION_TRANSFORMS.update(
    {
        task_name: lerobot_dataset_transform
        for task_name in LEROBOT_KITCHEN_NEW_TASKS
    }
)
