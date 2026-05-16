from utils.ignore_warning import ignore_warnings
ignore_warnings()

import json
import logging
import os
os.environ["MUJOCO_GL"] = "osmesa" # glfw(windowed) egl(GPU no-window) osmesa(CPU no-window)
import draccus
from dataclasses import dataclass
import random
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
import wandb

from libero.libero import benchmark
from peft import LoraConfig, get_peft_model

from config.config_vla import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from evaluation.action_sampling_utils import (
    get_vla_action,
    normalize_proprio,
    prepare_images_for_vla,
    process_action,
)
from evaluation.libero_utils import get_libero_env, prepare_observation
from finetune_ddp import RunConfig
from models.vla_model import MIRTH, MIRTHConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig(RunConfig):
    pretrained_vla_path: str             = "/mnt/data1/OpenVLA/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt" # Path to initiate model weights
    pretrained_checkpoint_path: str      = "/media/sunhao/T7/MemoryHubRunning/MBPrefixUnion_libero_goal_1T1A_VPHub_RToken_A6000x3/checkpoints/"      # Path to fine-tuned model checkpoint
    pretrained_checkpoint_step: int      = 30000                                                                                     # Steps of the checkpoint to load

    task_suite_name: str                 = None                       # Task suite to eval on
    device: int                          = 0
    
    num_trials_per_task: int             = 30                         # Number of trials to run per task
    num_steps_wait: int                  = 20                         # Number of steps to wait for objects to stabilize in sim
    env_img_res: int                     = 256                        # Resolution for environment images (not policy input resolution)
    model_img_res: int                   = 224                        # Resolution for model images
    center_crop: bool                    = True                       # Whether to center crop images from environment before resizing to model_img_res
    unnorm_keys: list                    = None                       # Keys to unnormalize in observation


TASK_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_configs_from_pretrained(config: EvalConfig):
    # Get checkpoint path
    all_checkpoints = os.listdir(config.pretrained_checkpoint_path)
    all_checkpoints = [ckpt for ckpt in all_checkpoints if ckpt.endswith(".pt") and 'latest' not in ckpt]
    all_steps = [int(ckpt.split("-")[1].split("-")[0]) for ckpt in all_checkpoints]
    assert config.pretrained_checkpoint_step in all_steps, f"Step {config.pretrained_checkpoint_step} not found in available checkpoints: {all_steps}"
        
    selected_ckpt = None
    for i in range(len(all_steps)):
        if all_steps[i] == config.pretrained_checkpoint_step:
            selected_ckpt = all_checkpoints[i]
            break
    if selected_ckpt is None:
        raise ValueError(f"Checkpoint for step {config.pretrained_checkpoint_step} not found.")
    pretrained_checkpoint = os.path.join(config.pretrained_checkpoint_path, selected_ckpt)
    
    # Update config
    config.pretrained_checkpoint_path = pretrained_checkpoint
    
    # Load pretrained config
    pretrained_config_path = os.path.join(os.path.dirname(os.path.dirname(config.pretrained_checkpoint_path)), "run_config.jsonl")
    assert os.path.exists(pretrained_config_path), f"Pretrained config file not found at {pretrained_config_path}"
    with open(pretrained_config_path, "r") as f:
        pretrained_config = json.load(f)['hparams']
    no_change_keys = ['pretrained_vla_path', 'pretrained_checkpoint_path', 'pretrained_checkpoint_step', 'run_dir']
    for key, value in pretrained_config.items():
        if key in no_change_keys:
            continue
        setattr(config, key, value)
        
    if config.use_original_action_tokens:
        config.action_token_type = "one_for_action_dim"    
    else:
        if config.one_token_for_action_chunk:
            config.action_token_type = "one_for_action_chunk"
        else:
            config.action_token_type = "one_for_action_step"
    
    if "_all_" in config.pretrained_checkpoint_path:
        assert config.task_suite_name is not None, "Please specify task_suite_name for _all_ checkpoints."
        if config.task_suite_name == "libero_object":
            config.unnorm_keys = "libero_object_no_noops"
        elif config.task_suite_name == "libero_spatial":
            config.unnorm_keys = "libero_spatial_no_noops"
        elif config.task_suite_name == "libero_goal":
            config.unnorm_keys = "libero_goal_no_noops"
        elif config.task_suite_name == "libero_10":
            config.unnorm_keys = "libero_10_no_noops"
        elif config.task_suite_name == "libero_90":
            config.unnorm_keys = "libero_90_no_noops"
        else:
            raise ValueError(f"Unknown task_suite_name: {config.task_suite_name}")
    else:
        if "_object_" in config.pretrained_checkpoint_path:
            setattr(config, 'task_suite_name', "libero_object")
        elif "_spatial_" in config.pretrained_checkpoint_path:
            setattr(config, 'task_suite_name', "libero_spatial")
        elif "_goal_" in config.pretrained_checkpoint_path:
            setattr(config, 'task_suite_name', "libero_goal")
        elif "_10_" in config.pretrained_checkpoint_path:
            setattr(config, 'task_suite_name', "libero_10")
        elif "_90_" in config.pretrained_checkpoint_path:
            setattr(config, 'task_suite_name', "libero_90")
        else:
            raise ValueError(f"Cannot infer task suite name from checkpoint path: {config.pretrained_checkpoint_path}")
        config.unnorm_keys = config.dataset_name
    print('--'*20)
    print('Task suite name:', config.task_suite_name)
    print('Pretrained dataset:', config.dataset_name)
    print('--'*20)
    
    return config


def initialize_model(config: EvalConfig, log_file):
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
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=min(config.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model.llm_backbone.llm = get_peft_model(model.llm_backbone.llm, lora_config)
    log_message("Model successfully initialized.", log_file)
    
    run_dir = Path(config.pretrained_checkpoint_path).parents[1]
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    with open(dataset_statistics_json, "r") as f:
        current_norm_stats = json.load(f)
        model.norm_stats = current_norm_stats

    weights_dict = torch.load(config.pretrained_checkpoint_path)['model']
    for module_key in weights_dict.keys():
        module = getattr(model, module_key)
        module.load_state_dict(weights_dict[module_key], strict=True)
        log_message("Successfully loaded " + module_key, log_file)
    log_message("Model weights successfully loaded.", log_file)
    model.eval()
    return model


def setup_logging(config: EvalConfig):
    os.makedirs(os.path.join(config.run_dir, config.run_id), exist_ok=True)
    local_log_filepath = os.path.join(config.run_dir, config.run_id, "eval_logs.txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    wandb.init(project=config.wandb_project, name=config.run_id+"_eval_"+str(config.pretrained_checkpoint_step)+'step', config=vars(config))
    return log_file, local_log_filepath


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    log_file.write(message + "\n")
    log_file.flush()


def save_rollout_video(run_dir, run_id, rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = os.path.join(run_dir, run_id, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(rollout_dir, f"episode={idx}--success={success}--task={processed_task_description}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def run_episode(
    config: EvalConfig,
    env,
    task_description: str,
    model,
    initial_state=None,
    log_file=None,
):
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    

    # Initialize history deques
    action_queue            = deque(maxlen=NUM_ACTIONS_CHUNK)
    action_normalized_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
    action_history          = deque(maxlen=config.history_window_size-1) # finaly union to [8, 9, 7] [batch, history, action_dim]
    pad_mask_history        = deque(maxlen=config.history_window_size)   # finaly union to [8, 9] [batch, history]
    proprio_history         = deque(maxlen=config.history_window_size-1) # finaly union to [8, 9, 9] [batch, history, proprio_dim]
    pixel_values_history    = deque(maxlen=config.history_window_size-1) # finaly union to pixel_values_return[dino][siglip] shape: (8, 2, 9, 3, 224, 224) [batch, num_camera, history, c, h, w]
    for _ in range(config.history_window_size - 1):
        action_history.append(np.zeros((1, 1, ACTION_DIM), dtype=np.float32))
        pad_mask_history.append(np.zeros((1, 1), dtype=bool))
        proprio_history.append(np.zeros((1, 1, PROPRIO_DIM), dtype=np.float32))
        pixel_values_history.append({
            'dino': torch.zeros((1, config.num_images_in_input, 1, 3, config.model_img_res, config.model_img_res), dtype=torch.float32),
            'siglip': torch.zeros((1, config.num_images_in_input, 1, 3, config.model_img_res, config.model_img_res), dtype=torch.float32),
        })
    pad_mask_history.append(np.ones((1, 1), dtype=bool))

    t, replay_images, success = 0, [], False
    max_steps = TASK_MAX_STEPS[config.task_suite_name]
    while t < max_steps + config.num_steps_wait + config.history_window_size:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < config.num_steps_wait:
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, config.model_img_res)
        replay_images.append(img)
        
        if len(action_queue) == 0:
            # Query model to get action
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    
                    action_history_tensor = torch.from_numpy(np.concatenate(np.array(action_history), axis=1)).to(config.device)           # [1, history, action_dim]
                    proprio_history_tensor = torch.from_numpy(np.concatenate(np.array(proprio_history), axis=1)).to(config.device)         # [1, history, proprio_dim]
                    pixel_values_history_tensor = {}
                    pixel_values_history_tensor['dino'] = torch.cat([pv['dino'] for pv in pixel_values_history], dim=2).to(config.device)     # [1, num_camera, history, c, h, w]
                    pixel_values_history_tensor['siglip'] = torch.cat([pv['siglip'] for pv in pixel_values_history], dim=2).to(config.device) # [1, num_camera, history, c, h, w]
                    pad_mask = torch.from_numpy(np.concatenate(np.array(pad_mask_history), axis=1)).to(config.device)  # [1, history+1]
                    
                    actions, normalized_actions, _, _ = get_vla_action(
                        config,
                        model,
                        observation,
                        task_description,
                        model.llm_backbone.prompt_builder,
                        action_history_tensor,
                        pixel_values_history_tensor,
                        proprio_history_tensor,
                        pad_mask,
                    )
            action_queue.extend(actions[0])
            action_normalized_queue.extend(normalized_actions[0])

        # Get action from queue
        action = action_queue.popleft()
        normalized_action = action_normalized_queue.popleft()

        # Process action
        action = process_action(action)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        
        # Update history deques
        action_history.append(normalized_action.reshape(1, 1, ACTION_DIM))
        # action_history.append(torch.zeros((1, 1, ACTION_DIM)))
        proprio_norm_stats = model.norm_stats[config.unnorm_keys]["proprio"]
        proprio = normalize_proprio(observation["state"], proprio_norm_stats)
        proprio_history.append(proprio.reshape(1, 1, PROPRIO_DIM))
        pad_mask_history.append(np.ones((1, 1), dtype=bool))
        
        all_images = [observation["full_image"]]
        if config.num_images_in_input > 1:
            all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
        assert len(all_images) == config.num_images_in_input, (f"Expected {config.num_images_in_input} images but got {len(all_images)}!")
        all_images = prepare_images_for_vla(all_images, config)
        all_images = [model.image_transform(image) for image in all_images] # [B, C, H, W]
        all_pixel_values = {
            'dino': torch.stack([v['dino'].unsqueeze(0).unsqueeze(1) for v in all_images], dim=1),       # [B, num_cameras, C, H, W]
            'siglip': torch.stack([v['siglip'].unsqueeze(0).unsqueeze(1) for v in all_images], dim=1),   # [B, num_cameras, C, H, W]
        }
        pixel_values_history.append({
            'dino': all_pixel_values['dino'].cpu(),
            'siglip': all_pixel_values['siglip'].cpu(),
        })

        if done:
            success = True
            break
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    config: EvalConfig,
    task_suite,
    task_id: int,
    model,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, resolution=config.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(config.num_trials_per_task)):
        log_message(f"\nTask: {task_description} Episode: {episode_idx + 1}", log_file)
        initial_state = initial_states[episode_idx]

        success, replay_images = run_episode(config, env, task_description, model, initial_state, log_file)

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(config.run_dir, config.run_id, replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file)
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    wandb.log({
        f"success_rate/{task_description}": task_success_rate,
        f"num_episodes/{task_description}": task_episodes,
    })

    return total_episodes, total_successes


def eval_libero(config: EvalConfig):
    # Set up seed and logging
    set_seed_everywhere(config.seed)
    log_file, local_log_filepath = setup_logging(config)
    device = torch.device('cuda:'+str(config.device))
    config.device = device

    # Initialize model
    model = initialize_model(config, log_file).to(config.device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config.task_suite_name]()
    num_tasks = task_suite.n_tasks
    log_message(f"Task suite: {config.task_suite_name}, Task number: {num_tasks}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(config, task_suite, task_id, model, total_episodes, total_successes, log_file)

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Final episodes: {total_episodes}", log_file)
    log_message(f"Final successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    wandb.log({
            "success_rate/total": final_success_rate,
            "success_episodes/total": total_successes,
            "num_episodes/total": total_episodes,
    })
    # Close local log file to ensure contents are flushed
    log_file.close()

    # Use W&B Artifacts instead of wandb.save to avoid potential hangs
    try:
        artifact = wandb.Artifact(name=f"{config.run_id}-eval-logs-step-{config.pretrained_checkpoint_step}", type="logs")
        artifact.add_file(local_log_filepath)
        wandb.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to log artifact to Weights & Biases: {e}")

    # Explicitly finish the W&B run
    try:
        wandb.finish()
    except Exception as e:
        logger.warning(f"Failed to finish Weights & Biases run: {e}")

    return final_success_rate


if __name__ == "__main__":
    # Use draccus to parse CLI args into EvalConfig and run.
    @draccus.wrap()
    def main(config: EvalConfig):
        print(f"Evaluating checkpoint at step {config.pretrained_checkpoint_step}...")
        config = get_configs_from_pretrained(config)
        eval_libero(config)

    main()