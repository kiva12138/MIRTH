"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from enum import IntEnum

from rlds_datasets.rlds.oxe.utils.droid_utils import zero_action_filter


# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    # fmt: on


# === Individual Dataset Configs ===
LEROBOT_KITCHEN_NEW_TASKS = tuple(f"task{i}" for i in range(1, 21))


def _make_lerobot_kitchen_new_config():
    return {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "front_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }


OXE_DATASET_CONFIGS = {
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_object_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}

OXE_DATASET_CONFIGS.update(
    {
        task_name: _make_lerobot_kitchen_new_config()
        for task_name in LEROBOT_KITCHEN_NEW_TASKS
    }
)
