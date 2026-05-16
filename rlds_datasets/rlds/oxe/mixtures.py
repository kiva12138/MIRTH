"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

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
    
    
    # LeRobot Datasets
    "lerobot_all": [
        ("level1_a", 1.0),
        ("level1_b", 1.0),
        ("level1_c", 1.0),
        ("level1_d", 1.0),
        ("level1_e", 1.0),
        ("level2_a", 1.0),
        ("level2_b", 1.0),
        ("level2_c", 1.0),
        ("level2_d", 1.0),
        ("level2_e", 1.0),
        ("level2_f", 1.0),
        ("level2_g", 1.0),
        ("level2_h", 1.0),
        ("level2_i", 1.0),
        ("level2_j", 1.0),
    ],
    "lerobot_a": [
        ("level1_a", 1.0),
        ("level1_b", 1.0),
        ("level1_c", 1.0),
        ("level1_d", 1.0),
        ("level1_e", 1.0),
    ],    
    "lerobot_b": [
        ("level2_a", 1.0),
        ("level2_b", 1.0),
        ("level2_c", 1.0),
        ("level2_d", 1.0),
        ("level2_e", 1.0),
        ("level2_f", 1.0),
        ("level2_g", 1.0),
        ("level2_h", 1.0),
        ("level2_i", 1.0),
        ("level2_j", 1.0),
    ],
    "basic_tasks": [
        ("level1_a", 1.0),
        ("level1_b", 1.0),
    ],
    "mechanism_ops": [
        ("level1_c", 1.0),
        ("level2_d", 1.0),
        ("level2_h", 1.0),
    ],
    "two_object_transfer_and_relative_pos": [
        ("level1_d", 1.0),
        ("level2_d", 1.0),
    ],
    "scene_rearrangement_and_alignment": [
        ("level1_e", 1.0),
        ("level1_b", 1.0),
        ("level2_i", 1.0),
        ("level2_e", 1.0),
    ],
    "category_attribute_grouping": [
        ("level2_a", 1.0),
        ("level2_f", 1.0),
        ("level2_e", 1.0),
        ("level2_i", 1.0),
    ],
    "state_based_inference_cut_uncut_visibility": [
        ("level2_b", 1.0),
        ("level2_j", 1.0),
    ],
    "recipe_semantic_composition": [
        ("level2_c", 1.0),
        ("level2_g", 1.0),
    ],
    "tidying_storage_clearing": [
        ("level2_e", 1.0),
        ("level2_i", 1.0),
        ("level2_a", 1.0),
    ],
    "count_condition_order_constraints": [
        ("level2_j", 1.0),
        ("level2_b", 1.0),
    ],

    # New LeRobot Dataset
    "new_lerobot_1":[
        ("place_the_banana_in_the_plate_on_the_right", 1.0),
        ("place_the_brown_kiwi_on_the_cutting_board", 1.0),
        ("place_the_carrot_in_the_plate_on_the_left", 1.0),
        ("place_the_star_fruit_in_the_white_frying_pan", 1.0),
    ],
    "new_lerobot_2":[
        ("close_the_second_drawer_of_the_four_drawer_cabinet", 1.0),
        ("open_the_second_drawer_put_the_banana_into_it_and_close_the_drawer", 1.0),
        ("open_the_top_drawer_of_the_four_drawer_cabinet", 1.0),
        ("open_the_top_drawer_place_the_spatula_inside_it_and_close_the_drawer", 1.0),
    ]
}
# fmt: on
