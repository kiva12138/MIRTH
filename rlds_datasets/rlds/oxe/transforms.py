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
    traj_len = tf.shape(trajectory["action"])[0]

    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["EEF_state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["gripper_state"] = tf.zeros((traj_len, 1), dtype=tf.float32)
    return trajectory



# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "level1_a": lerobot_dataset_transform,
    "level1_b": lerobot_dataset_transform,
    "level1_c": lerobot_dataset_transform,
    "level1_d": lerobot_dataset_transform,
    "level1_e": lerobot_dataset_transform,
    "level2_a": lerobot_dataset_transform,
    "level2_b": lerobot_dataset_transform,
    "level2_c": lerobot_dataset_transform,
    "level2_d": lerobot_dataset_transform,
    "level2_e": lerobot_dataset_transform,
    "level2_f": lerobot_dataset_transform,
    "level2_g": lerobot_dataset_transform,
    "level2_h": lerobot_dataset_transform,
    "level2_i": lerobot_dataset_transform,
    "level2_j": lerobot_dataset_transform,
    "place_the_banana_in_the_plate_on_the_right": lerobot_dataset_transform,
    "place_the_brown_kiwi_on_the_cutting_board": lerobot_dataset_transform,
    "place_the_carrot_in_the_plate_on_the_left": lerobot_dataset_transform,
    "place_the_star_fruit_in_the_white_frying_pan": lerobot_dataset_transform,
    "close_the_second_drawer_of_the_four_drawer_cabinet": lerobot_dataset_transform,
    "open_the_second_drawer_put_the_banana_into_it_and_close_the_drawer": lerobot_dataset_transform,
    "open_the_top_drawer_of_the_four_drawer_cabinet": lerobot_dataset_transform,
    "open_the_top_drawer_place_the_spatula_inside_it_and_close_the_drawer": lerobot_dataset_transform,

}
