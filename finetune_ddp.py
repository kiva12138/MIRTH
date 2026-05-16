"""
echo 1 | sudo tee /proc/sys/vm/drop_caches
NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=1 torchrun --standalone --nnodes 1 --nproc-per-node 3 finetune_ddp.py
"""
from utils.ignore_warning import ignore_warnings
ignore_warnings()

import os
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import draccus
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

from models.vla_model import MIRTHConfig, MIRTHOutput, MIRTH
from utils.overwatch import initialize_overwatch
from utils.metrics import Metrics
from utils.torch_utils import set_global_seed, worker_init_function
from utils.data_utils import PaddedCollatorForActionPrediction, save_dataset_statistics, as_float, get_memory_usage
from rlds_datasets import RLDSBatchTransform, RLDSDataset
from peft import LoraConfig, get_peft_model

overwatch = initialize_overwatch(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class RunConfig:
    # Path related
    pretrained_vla_path: str             = "/mnt/data1/OpenVLA/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
    hf_token: str                        = "/path/to/hf_token"                             # Hugging Face token path
    data_root_dir: str                   = "/path/to/LIBERO/modified_libero_rlds/"         # Directory containing RLDS datasets
    run_dir: str                         = "/path/to/MemoryHubRunning"                     # Directory to store logs & checkpoints
    dataset_name: str                    = "libero_goal_no_noops"                          # Name of fine-tuning dataset
    run_id: str                          = "MBPrefixUnion_libero_goal_1T1A_VPHub_RToken_A6000x3"
    
    # Model
    num_images_in_input: int             = 2                                              # Number of images provided as input
    stage: str                           = "lvp"                                          # Stage for the model to freeze backbones; options: ["lvp", "lv"]
    action_token_type: str                 = "one_for_action_step"                          # How to represent action tokens in the input prompt; options: ["one_for_action_chunk", "one_for_action_step", "one_for_action_dim"]
    use_proprio: bool                    = True                                           # Use proprioceptive information
    use_timestamp: bool                  = False                                          # Use timestamp information in action token initialization
    action_biattnn: bool                 = False                                          # Use bi-attention for action tokens
    use_contrastive_loss: bool           = False                                          # Use contrastive loss
    contrastive_tau_ra: float            = 0.07                                           # Temperature ratio for contrastive loss, 0.07 and 0.1 are most commonly used
    contrastive_tau_rx: float            = 0.07                                           # Temperature ratio for contrastive loss, 0.07 and 0.1 are most commonly used
    lambda_contrastive_ra: float         = 0.001                                          # Weight for contrastive loss
    lambda_contrastive_rx: float         = 0.001                                          # Weight for contrastive loss
    
    # Parameters for Memory hub
    mb_prefix_type : str                 = "union"                                        # union | separate
    use_vision_memory_hub: bool          = True                                           # Enable vision memory hub
    use_proprio_memory_hub: bool         = True                                           # Enable proprio memory hub
    use_action_memory_hub: bool          = False                                          # Enable action memory hub
    long_memory_scale_number: int        = 4                                              # Scale factor for long memory
    short_memory_length: int             = 4                                              # Length of short-term memory
    tau: float                           = 1.0                                            # similarity temperature
    beta_min: float                      = 0.01                                           # Minimum beta
    beta_max: float                      = 0.3                                            # Maximum beta
    gamma: float                         = 0.2                                            # Gamma parameter
    lmbd: float                          = 0.2                                            # Lambda regularization weight
    bias: float                          = 1.0                                            # Bias term
    
    # Parameters for action reason token
    use_reason_token: bool               = True
    num_reason_token: int                = 4                                              # Number of reason tokens
    reason_hidden: int                   = 128                                            # Hidden size for reason module
    reason_p_drop: float                 = 0.0                                            # Dropout for reason module
    reason_out_scale: float              = 1.0                                            # Output scaling for reason module

    # Dataset and Logging
    image_aug: bool                   = True                                            # If True, enable image augmentations
    shuffle_buffer_size: int          = 10_000                                          # Dataloader shuffle buffer size (reduce if OOM)
    wandb_project: str                = "OpenVLAOFT-My"                                 # WandB project name
    active_trackers: tuple            = ("jsonl", "wandb")                              # Active trackers for logging metrics
    window_size: int                  = 128                                             # Logging window size
    history_window_size               = 20                                              # History window size for metrics

    # Training configuration
    global_batch_size: int            = 64                                              # Global (effective) batch size
    per_device_batch_size: int        = 32                                              # Batch size per device
    learning_rate: float              = 5e-4                                            # Base learning rate
    weight_decay: float               = 0.0                                             # Weight decay for optimizer
    max_grad_norm: float              = 0.5                                             # Max gradient norm for clipping
    lr_scheduler_type: str            = "linear-warmup+cosine-decay"                    # Learning rate scheduler type (linear-warmup+cosine-decay or constant)
    warmup_ratio: float               = 0.001                                           # Warmup ratio for LR scheduler
    epochs: int                       = 10                                              # Number of training epochs
    max_steps: int                    = 80_000                                          # Max training steps
    save_freq: int                    = 2_000                                           # Checkpoint saving frequency (in steps)
    resume: bool                      = False                                           # If True, resume from checkpoint
    reduce_in_full_precision: bool    = False                                           # If True, reduce gradients in full precision
    seed: int                         = 7

    # LoRA (low-rank adaptation) settings
    use_lora: bool                    = True                                            # Enable LoRA fine-tuning
    lora_rank: int                    = 32                                              # LoRA rank
    lora_dropout: float               = 0.0                                             # Dropout applied to LoRA weights



class DDPStrategy:
    def __init__(self, model: MIRTH, metrics: Metrics, device_id: int, config: RunConfig):
        self.model, self.device_id = model, device_id
        self.config = config
        self.metrics = metrics

        self.optimizer, self.lr_scheduler = None, None

        assert (self.config.global_batch_size % self.config.per_device_batch_size == 0), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.config.global_batch_size // self.config.per_device_batch_size // overwatch.world_size()

    def run_training(
        self,
        dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: Metrics,
        save_full_model: bool = True,
    ) -> None:
        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=worker_init_function,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=((self.config.epochs * (len(dataloader) // self.grad_accumulation_steps)) if self.config.max_steps is None else self.config.max_steps),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.model.train()
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            accum_curr_action_l1_loss, accum_next_actions_l1_loss, accum_all_action_l1_loss = 0.0, 0.0, 0.0
            accum_contrastive_loss, accum_contrastive_loss_ra, accum_contrastive_loss_rx = 0.0, 0.0, 0.0
            accum_micro_batches = 0
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # [Contract] self.model.forward() must automatically compute `loss` and return!
                    output: MIRTHOutput = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        current_action=batch["current_action"],
                        current_action_chunk=batch["current_action_chunk"],
                        history_actions=batch['history_actions'],
                        pixel_values=batch['pixel_values'],
                        pixel_values_history=batch['pixel_values_history'],
                        proprio=batch["proprio"],
                        proprio_history=batch["proprio_history"],
                        pad_mask=batch["pad_mask"],
                        debug=False
                    )
                    loss = output.loss
                    metrics_return = output.metrics

                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()
                
                # # === Gradient check: print grad norms for all trainable parameters ===
                # if overwatch.is_rank_zero():
                #     for name, param in self.model.named_parameters():
                #         if param.requires_grad:
                #             grad_norm = param.grad.data.norm().item()
                #             print(f"[GradCheck] Step {metrics.global_step} | {name}: grad_norm={grad_norm:.6f} NotNone {param.grad is not None}")
                #             assert param.grad is not None, f"Gradient for parameter {name} is None!"
                # # ================================================================

                accum_curr_action_l1_loss  += as_float(metrics_return.get('current_action_l1_loss', 0.0))
                accum_next_actions_l1_loss += as_float(metrics_return.get('next_actions_l1_loss', 0.0))
                accum_all_action_l1_loss   += as_float(loss)
                accum_contrastive_loss     += as_float(metrics_return.get('contrastive_loss', 0.0))
                accum_contrastive_loss_ra  += as_float(metrics_return.get('contrastive_loss_ra', 0.0))
                accum_contrastive_loss_rx  += as_float(metrics_return.get('contrastive_loss_rx', 0.0))
                accum_micro_batches        += 1

                if (accum_micro_batches % self.grad_accumulation_steps) == 0:
                    
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                    avg_curr_action_l1_loss  = float(accum_curr_action_l1_loss / accum_micro_batches)
                    avg_next_actions_l1_loss = float(accum_next_actions_l1_loss / accum_micro_batches)
                    avg_all_action_l1_loss   = float(accum_all_action_l1_loss / accum_micro_batches)
                    avg_contrastive_loss     = float(accum_contrastive_loss / accum_micro_batches)
                    avg_contrastive_loss_ra  = float(accum_contrastive_loss_ra / accum_micro_batches)
                    avg_contrastive_loss_rx  = float(accum_contrastive_loss_rx / accum_micro_batches)

                    # Get memory usage
                    mem_stats = get_memory_usage()

                    metrics.commit(loss=avg_all_action_l1_loss)
                    metrics.commit(current_action_l1_loss=avg_curr_action_l1_loss, next_actions_l1_loss=avg_next_actions_l1_loss)
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                    metrics.commit(**mem_stats)  # Add memory stats
                    metrics.commit(contrastive_loss=avg_contrastive_loss, contrastive_loss_ra=avg_contrastive_loss_ra, contrastive_loss_rx=avg_contrastive_loss_rx)
                    metrics.commit_time()
                    status = metrics.push()

                    batches_per_epoch = len(dataset) // self.config.global_batch_size
                    epoch = 0 if batches_per_epoch == 0 else (metrics.global_step + 1) // batches_per_epoch

                    if (terminate := (self.config.max_steps is not None and metrics.global_step >= self.config.max_steps)) or ((metrics.global_step % self.config.save_freq) == 0):
                        self.save_checkpoint(metrics.run_dir, metrics.run_id, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model)
                        dist.barrier()

                        if terminate:
                            return

                    progress.update()
                    progress.set_description(status)

                    accum_curr_action_l1_loss, accum_next_actions_l1_loss, accum_all_action_l1_loss = 0.0, 0.0, 0.0
                    accum_contrastive_loss, accum_contrastive_loss_ra, accum_contrastive_loss_rx = 0.0, 0.0, 0.0
                    accum_micro_batches = 0
                
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        run_id: str,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint containing (optionally) only the trainable parameters."""
        assert isinstance(self.model, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        model_state_dicts = {}
        for name, submod in self.model.module.named_children():
            if only_trainable:
                # include submodule only if it has any trainable parameters
                if not any(p.requires_grad for p in submod.parameters()):
                    continue
            model_state_dicts[name] = submod.state_dict()

        # Fallback: if no top-level children were added (e.g., single-module model),
        # store the full module state_dict under the key "model".
        if not model_state_dicts:
            model_state_dicts["model"] = self.model.module.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        checkpoint_dir = Path(run_dir) / Path(run_id) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, n_train_examples: int) -> None:
        self.model.llm_backbone.enable_gradient_checkpointing()
        # self.model.llm_backbone.disable_gradient_checkpointing()

        overwatch.info("Placing Entire Model on GPU", ctx_level=1)
        self.model.to(self.device_id)

        overwatch.info("Wrapping Model with Distributed Data Parallel", ctx_level=1)
            
        self.model = DDP(self.model, device_ids=[self.device_id], gradient_as_bucket_view=True, find_unused_parameters=False)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        overwatch.info("Wrapped Model with Distributed Data Parallel", ctx_level=1)
        
        if self.config.max_steps is None:
            num_training_steps = (n_train_examples * self.config.epochs) // self.config.global_batch_size
        else:
            num_training_steps = self.config.max_steps

        if self.config.lr_scheduler_type == "linear-warmup+cosine-decay":
            num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            assert self.config.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0
        elif self.config.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.config.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.config.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.config.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> Default AdamW LR = {self.config.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.config.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.config.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.config.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)


def train(config = RunConfig()) -> None:
    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    worker_init_fn = set_global_seed(config.seed, get_worker_init_fn=True)
    os.makedirs(Path(config.run_dir) / Path(config.run_id), exist_ok=True)
    os.makedirs(Path(config.run_dir) / Path(config.run_id) / "checkpoints", exist_ok=True)

    model_config = MIRTHConfig(
        pretrained_vla_path=config.pretrained_vla_path,
        num_images_in_input=config.num_images_in_input,
        use_proprio=config.use_proprio,
        hf_token=config.hf_token,
        action_token_type=config.action_token_type,
        mb_prefix_type = config.mb_prefix_type,
        use_vision_memory_hub=config.use_vision_memory_hub,
        use_proprio_memory_hub=config.use_proprio_memory_hub,
        use_action_memory_hub=config.use_action_memory_hub,
        long_memory_scale_number=config.long_memory_scale_number,
        short_memory_length=config.short_memory_length,
        tau=config.tau,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        gamma=config.gamma,
        lmbd=config.lmbd,
        bias=config.bias,
        use_reason_token=config.use_reason_token,
        num_reason_token=config.num_reason_token,
        reason_hidden=config.reason_hidden,
        reason_p_drop=config.reason_p_drop,
        reason_out_scale=config.reason_out_scale,
        use_timestamp = config.use_timestamp,
        action_biattnn = config.action_biattnn,
        use_contrastive_loss = config.use_contrastive_loss,
        contrastive_tau_ra = config.contrastive_tau_ra,
        contrastive_tau_rx = config.contrastive_tau_rx,
        lambda_contrastive_ra = config.lambda_contrastive_ra,
        lambda_contrastive_rx = config.lambda_contrastive_rx,
    )
    model = MIRTH(model_config)
    overwatch.info(f"Training Stage: `{config.stage}`")
    model.freeze_backbones(config.stage)
    if config.use_lora:
        overwatch.info(f"Wrapping model with LoRA (rank={config.lora_rank}, dropout={config.lora_dropout})")
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=min(config.lora_rank, 16),
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model.llm_backbone.llm = get_peft_model(model.llm_backbone.llm, lora_config)
    
    if config.resume:
        checkpoint_path = Path(config.run_dir) / Path(config.run_id) / Path("checkpoints") / Path("latest-checkpoint.pt")
        if not checkpoint_path.exists():
            overwatch.warn(f"Checkpoint not found at {checkpoint_path}, skipping resume load")
        else:
            overwatch.info(f"Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model_states = ckpt.get("model", ckpt)

            # If checkpoint contains a single full state_dict under key "model"
            if isinstance(model_states, dict) and "model" in model_states and isinstance(model_states["model"], dict):
                full_state = model_states["model"]
                overwritten, missing = model.load_state_dict(full_state, strict=False)
                overwatch.info(f"Loaded full model state_dict (missing keys: {len(missing)}, overwritten: {len(overwritten)})")
            else:
                # Otherwise we expect a mapping of top-level submodule name -> state_dict
                named_children = dict(model.named_children())
                loaded_subs = 0
                for sub_name, sub_state in model_states.items():
                    submod = named_children.get(sub_name)
                    if submod is None:
                        overwatch.warn(f"Checkpoint contains submodule '{sub_name}' which is not present in the current model; skipping")
                        continue
                    submod.load_state_dict(sub_state, strict=False)
                    loaded_subs += 1
                overwatch.info(f"Loaded {loaded_subs} submodule state_dict(s) from checkpoint")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    overwatch.info(f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable")
    
    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{config.dataset_name}`")
    batch_transform = RLDSBatchTransform(model.llm_backbone.tokenizer, 
                                         model.image_transform, 
                                         model.llm_backbone.prompt_builder, 
                                         action_token_type=config.action_token_type,
                                         num_reason_tokens=config.num_reason_token,
                                         use_reason_token=config.use_reason_token
                                         )
    dataset = RLDSDataset(
        data_root_dir=config.data_root_dir,
        data_mix=config.dataset_name,
        batch_transform = batch_transform,
        resize_resolution = (model.vision_backbone.vision_backbone.default_image_size, model.vision_backbone.vision_backbone.default_image_size),
        shuffle_buffer_size = config.shuffle_buffer_size,
        load_proprio = config.use_proprio,
        load_camera_views=("primary", "wrist") if config.num_images_in_input == 2 else ("primary",),
        train = True,
        image_aug = config.image_aug,
        history_window_size = config.history_window_size,
    )
    tokenizer = model.llm_backbone.tokenizer
    collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right")
    
    if overwatch.is_rank_zero():
        save_dataset_statistics(dataset.dataset_statistics, Path(config.run_dir) / Path(config.run_id))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{config.active_trackers}`")
    metrics = Metrics(
        config.active_trackers,
        config.run_id,
        config.run_dir,
        draccus.encode(config),
        wandb_project=config.wandb_project,
        grad_accumulation_steps=config.global_batch_size // config.per_device_batch_size // overwatch.world_size(),
        window_size=config.window_size,
    )
    
    # Create Train Strategy
    overwatch.info(f"Initializing DDP Train Strategy")
    # torch.distributed.barrier()  # Wait for all processes to be ready before DDP(...)
    
    with open("./parameters.txt", "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\t{param.shape}\t{param.requires_grad}\n")
    train_strategy = DDPStrategy(model=model, metrics=metrics, device_id=device_id, config=config)
    train_strategy.run_setup(n_train_examples=len(dataset))

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(dataset=dataset, collator=collator, metrics=metrics, save_full_model=False)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()