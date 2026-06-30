"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

from rlds_datasets.rlds.oxe.configs import LEROBOT_KITCHEN_NEW_TASKS

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === LIBERO Datasets (Modified Versions) ===
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
    "libero_all": [
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_goal_no_noops", 1.0),
        ("libero_10_no_noops", 1.0),
    ],

    # === LeRobotKitchenNew RLDS Datasets ===
    "lerobot_kitchen_new": [(task_name, 1.0) for task_name in LEROBOT_KITCHEN_NEW_TASKS],
    "lerobot_kitchen_new_basic_tasks": [
        ("task1", 1.0),
        ("task2", 1.0),
        ("task3", 1.0),
        ("task4", 1.0),
    ],
    "lerobot_kitchen_new_category_reasoning": [
        ("task5", 1.0),
        ("task6", 1.0),
        ("task7", 1.0),
        ("task8", 1.0),
    ],
    "lerobot_kitchen_new_mechanism_operations": [
        ("task9", 1.0),
        ("task10", 1.0),
        ("task11", 1.0),
        ("task12", 1.0),
    ],
    "lerobot_kitchen_new_scene_rearrange": [
        ("task13", 1.0),
        ("task14", 1.0),
        ("task15", 1.0),
        ("task16", 1.0),
    ],
    "lerobot_kitchen_new_semantic_recipe": [
        ("task17", 1.0),
        ("task18", 1.0),
        ("task19", 1.0),
        ("task20", 1.0),
    ],
    
}
# fmt: on
