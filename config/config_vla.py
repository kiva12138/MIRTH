from enum import Enum

# ROBOT_PLATFORM = "LIBERO"
ROBOT_PLATFORM = "LEROBOT"

# ======== Constants for VLM ========
IGNORE_INDEX = -100
STOP_INDEX = 2  # '</s>'
MAX_TOKEN_INDEX = 31999
SPECIAL_TOKEN_IDX_AT_END = 29871

# ======== VLA-specific Constants ========
ACTION_TOKEN_BEGIN_IDX = 31743           # Where the action tokens start = MAX_TOKEN_INDEX - ACTION_BINS
ACTION_TOKEN_END_IDX = MAX_TOKEN_INDEX   # Where the action tokens end
ACTION_BINS = 256 # The action tokens will be with interval ACTION_TOKEN_BEGIN_IDX <= action <= ACTION_TOKEN_BEGIN_IDX + ACTION_BINS

# ========== MIRTH's Optimized Constants ========
SINGLE_ACTION_TOKEN_INDEX = MAX_TOKEN_INDEX              # For scenarios where we sample one complete action by one single token
SINGLE_ACTION_CHUNK_TOKEN_INDEX = MAX_TOKEN_INDEX - 1    # For scenarios where we sample one complete action chunk by one single token
ACTION_REASON_TOKEN_BEGIN_IDX = 31679
ACTION_REASON_TOKEN_END_IDX = 31742
ACTION_REASON_TOKEN_NUMBER = 64

# ======== Action Normalization Constants ========
class NormalizationType(str, Enum):
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]

# ======== Robot Platform Specific Constants ========
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 9, # Originally 8, but I don't know why we have 9 dimensions for LIBERO proprio, but this will work for now since we are only using the 8 dimensions for proprio
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LEROBOT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 15,
    "ACTION_DIM": 6,
    "PROPRIO_DIM": 6,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# ========= Select Constants Based on Robot Platform ========
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "LEROBOT":
    constants = LEROBOT_CONSTANTS
    
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

