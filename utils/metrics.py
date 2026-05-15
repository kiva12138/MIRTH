"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""
import warnings
warnings.filterwarnings("ignore", message=r"The 'repr' attribute *", category=UserWarning, module=r"pydantic\._internal\._generate_schema",)
warnings.filterwarnings("ignore", message=r"The 'frozen' attribute *", category=UserWarning, module=r"pydantic\._internal\._generate_schema",)


import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

import jsonlines
import numpy as np
import torch
import wandb

from utils.overwatch import initialize_overwatch
from utils.data_utils import as_float

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(Path(self.run_dir) / self.run_id / "run_config.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(Path(self.run_dir) / self.run_id / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return


class WeightsBiasesTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "Test",
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams
        self.project, self.wandb_dir = project, self.run_dir

        # Call W&B.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)


class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        wandb_project: str = "prismatic",
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams
        self.window_size = window_size

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(run_id, run_dir, hparams, project=wandb_project)
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss_recent": deque(maxlen=grad_accumulation_steps),
            "loss_smoothed": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": deque(maxlen=1),
        }

    def get_status(self) -> str:
        lr = self.state["lr"][-1] if self.state["lr"] else 0.0
        loss = torch.Tensor(list(self.state["loss_smoothed"])).mean().item() if self.state["loss_smoothed"] else 0.0
        return f"=>> [Global Step] {self.global_step:06d} =>> LR: {lr:.6f} -- Loss: {loss:.4f}"

    def commit(self, *, global_step: Optional[int] = None, lr: Optional[float] = None, **kwargs):
        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero:
            return
        
        if global_step is not None:
            self.global_step = global_step

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = as_float(value.detach()) if isinstance(value, torch.Tensor) else as_float(value)
                self.state["loss_recent"].append(loss_val)
                self.state["loss_smoothed"].append(loss_val)
                continue

            if key not in self.state:
                self.state[key] = deque(maxlen=self.window_size)

            metric_value = as_float(value.detach()) if isinstance(value, torch.Tensor) else as_float(value)
            self.state[key].append(metric_value)
        
    def commit_time(self):
        if not overwatch.is_rank_zero:
            return
        self.state["step_time"].append(time.time() - self.step_start_time)
        self.step_start_time = time.time()

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        def _mean_from_buffer(buffer_values):
            if not buffer_values:
                return 0.0

            first_item = buffer_values[0]
            if isinstance(first_item, torch.Tensor):
                stacked = torch.stack([
                    item if isinstance(item, torch.Tensor) else torch.tensor(item)
                    for item in buffer_values
                ])
                return stacked.mean().item()

            return float(np.mean(buffer_values))

        loss_recent = _mean_from_buffer(list(self.state["loss_recent"]))
        loss_smoothed = _mean_from_buffer(list(self.state["loss_smoothed"]))
        step_time = _mean_from_buffer(list(self.state["step_time"]))

        lr_buffer = list(self.state["lr"])
        lr = lr_buffer[-1] if lr_buffer else 0.0

        metrics_payload = {
            "loss_smoothed": loss_smoothed,
            "loss_recent": loss_recent,
            "lr": lr,
            "step_time": step_time,
        }

        base_keys = list(metrics_payload.keys() )
        for key, buffer in self.state.items():
            if key in base_keys:
                continue

            buffer_values = list(buffer)
            if not buffer_values:
                continue

            metrics_payload[key] = _mean_from_buffer(buffer_values)

        # Fire to Trackers
        for tracker in self.trackers:
            tracker.write(self.global_step, metrics_payload)

        return self.get_status()

    def finalize(self) -> str:
        for tracker in self.trackers:
            tracker.finalize()

