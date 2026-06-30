# MIRTH

The codes for **MIRTH: Mutual-Information Reasoning with Temporal Hubs for Vision-Language-Action Agents**.

> Caution: The accepted version of the paper contains errors regarding symbols and repetitions; please refer to this latest version uploaded to arXiv (here goes the link).

> Still updating. This repository will be made fully public soon.

---

## 1. Installation

This project is built upon OpenVLA, but we use a relatively new toolchain (PyTorch 2.9 + CUDA 13) for better extendability with future VLA / VLM stacks. Older environments may also work, but require more configurations.

### 1.1 Create a fresh environment

```bash
conda create -n mirth python=3.13 -y
conda activate mirth
```

### 1.2 Install requirements

```bash
pip install -r requirements.txt
pip install --no-deps "dlimp @ https://codeload.github.com/moojink/dlimp_openvla/zip/refs/heads/main#sha256=e3140251551630c58fe935ef553bdad7e856bb028e8721155d00da714045f665"
```

The pinned versions in [requirements.txt](requirements.txt) include `torch==2.9.1+cu130`, `torchvision==0.24.1+cu130`, `transformers==4.57.3`, `tensorflow==2.20.0`, `tensorflow-datasets==4.9.9`, and `peft==0.18.0`. The file includes the PyTorch CUDA 13.0 wheel index. Make sure your CUDA driver supports CUDA 13. If you must stay on an older CUDA, install the matching `torch` / `torchvision` wheels first, then run the rest of `requirements.txt`.

`dlimp` is installed separately with `--no-deps` because its package metadata pins `tensorflow==2.15.0`, which is not available for Python 3.13. MIRTH uses the TensorFlow 2.20 stack pinned above.

LIBERO is installed as an editable git package from `requirements.txt`. If the editable install fails on your machine, install it manually:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
```

### 1.3 Install Flash Attention (recommended)

We strongly recommend installing [`flash-attn`](https://github.com/Dao-AILab/flash-attention) — training is significantly faster and memory consumption is much lower:

```bash
pip install flash_attn==2.8.3 --no-build-isolation
```

If the prebuilt wheel is unavailable for your platform, pip will try to build `flash-attn` from source. This requires a full CUDA Toolkit installation, not just the PyTorch CUDA wheel. Before building, `nvcc --version` must work and `CUDA_HOME` / `CUDA_PATH` must point to the CUDA install root. On Windows, source builds are especially fragile; if you do not already have a working CUDA compiler toolchain, use the SDPA fallback below.

**If you cannot install `flash_attn`**, you need to fall back to PyTorch's SDPA / math kernels:

1. In [models/vla_model.py:90](models/vla_model.py#L90), change `use_flash_attention_2=True` to `use_flash_attention_2=False`. This makes [models/llm_llama2.py:72](models/llm_llama2.py#L72) and [models/llm_llama2.py:77](models/llm_llama2.py#L77) use `attn_implementation="sdpa"` instead.
2. In [finetune_ddp.py:37-40](finetune_ddp.py#L37-L40), uncomment the four `torch.backends.cuda.enable_*_sdp(...)` lines so that the math-based SDP kernel is enabled as a fallback:

   ```python
   torch.backends.cuda.enable_mem_efficient_sdp(False)
   torch.backends.cuda.enable_cudnn_sdp(True)
   torch.backends.cuda.enable_flash_sdp(True)
   torch.backends.cuda.enable_math_sdp(True)
   ```

Note that the non-flash path is **noticeably slower (about 0.3x)** and uses more GPU memory; we recommend resolving the `flash_attn` build (e.g. by upgrading PyTorch / CUDA) rather than running without it.

---

## 2. Downloads

Download the pretrained OpenVLA-7B (Prismatic) checkpoint from [openvla/openvla-7b-prismatic](https://huggingface.co/openvla/openvla-7b-prismatic), then set `pretrained_vla_path` / `PRETRAINED_VLA_PATH` to the local `.pt` checkpoint path.

Baidu Disk: https://pan.baidu.com/s/1d8RFeruwF5124L2t4BFUkg?pwd=7890 Code: 7890

Google Drive: https://drive.google.com/drive/folders/12B_y0w7uoEtVVO91aHMqPuNtV2fbYXGs?usp=drive_link

MIRTH supports two dataset formats:

| Format | Loader | Download link |
| --- | --- | --- |
| RLDS / TFDS format | `rlds_datasets.RLDSDataset` | TBD |
| LeRobot format | `lerobot_datasets.LeRobotOpenVLADataset` | TBD |

Both loaders adapt samples to the same OpenVLA-style batch contract before collation, so they can share `PaddedCollatorForActionPrediction`.

---

## 3. Smoke-test the environment

Before launching a multi-GPU fine-tune, run the `Test*.py` scripts in the repository root to verify that each major component loads correctly on your machine. They are small, self-contained, and surface most installation issues (missing CUDA libs, broken `flash_attn`, wrong `transformers` version, bad HF token, missing RLDS data, etc.) much faster than a full training run.

Before running these scripts, edit their path variables so they point to your real downloaded files and dataset roots, including the OpenVLA checkpoint path, Hugging Face token file, and RLDS / LeRobot dataset directories.

Run them one by one:

| Script | What it checks |
| --- | --- |
| [TestVisionEncoders.py](TestVisionEncoders.py) | Loads the Prism DINO + SigLIP vision backbone and the `InfusedDinoSigLIPViTBackbone` wrapper, runs a dummy forward pass on the GPU. |
| [TestLLM.py](TestLLM.py) | Loads the pretrained LLaMA-2 backbone via Hugging Face, runs a forward pass on a templated prompt. **Edit `hf_token`** at the top of the file before running. |
| [TestMemoryHub.py](TestMemoryHub.py) | Builds `VisionMemoryHubForTraining` / `ProprioMemoryHubForTraining` with synthetic inputs end-to-end through the infused vision encoder. |
| [TestVLA.py](TestVLA.py) | Instantiates the full `MIRTH` model with a synthetic batch — the closest single-process proxy for what `finetune_ddp.py` does. |
| [TestDataset.py](TestDataset.py) | Builds an `RLDSDataset` + `PaddedCollatorForActionPrediction` and iterates a few batches. **Edit `DATA_ROOT_DIR`, `DATASET_NAME`, `HF_TOKEN`** at the top of the file. Use this to confirm your RLDS data is in the expected location and format. |

Typical invocation:

```bash
python TestVisionEncoders.py
python TestLLM.py
python TestMemoryHub.py
python TestVLA.py
python TestDataset.py
```

If all five pass, your environment is ready for fine-tuning.

---

## 4. Fine-tuning

### 4.1 Configure paths and hyperparameters

All training options live in the `RunConfig` dataclass at the top of [finetune_ddp.py](finetune_ddp.py). Before launching, edit at least the following fields to match your environment:

| Field | Meaning |
| --- | --- |
| `pretrained_vla_path` | Path to the pretrained OpenVLA-7B (Prismatic) checkpoint `.pt` file downloaded from [openvla/openvla-7b-prismatic](https://huggingface.co/openvla/openvla-7b-prismatic) |
| `hf_token` | Path to a file containing your Hugging Face access token |
| `data_root_dir` | Directory containing the training datasets; use the RLDS / TFDS root for `RLDSDataset`, or the LeRobot root for `LeRobotOpenVLADataset` |
| `run_dir` | Where logs and checkpoints will be written |
| `dataset_name` | RLDS mixture name, e.g. `libero_goal_no_noops`, `libero_object_no_noops`, `libero_spatial_no_noops`, `libero_10_no_noops` |
| `run_id` | A unique identifier for this run (used as the checkpoint subdirectory) |

Other commonly tuned fields:

- **Memory hub**: `use_vision_memory_hub`, `use_proprio_memory_hub`, `use_action_memory_hub`, `mb_prefix_type` (`union` / `separate`), `long_memory_scale_number`, `short_memory_length`, and the `tau / beta_min / beta_max / gamma / lmbd / bias` weighting parameters.
- **Reason tokens**: `use_reason_token`, `num_reason_token`, `reason_hidden`, `reason_p_drop`, `reason_out_scale`.
- **Contrastive loss** (optional): `use_contrastive_loss`, `contrastive_tau_ra`, `contrastive_tau_rx`, `lambda_contrastive_ra`, `lambda_contrastive_rx`.
- **Training schedule**: `global_batch_size`, `per_device_batch_size`, `learning_rate`, `epochs`, `max_steps`, `save_freq`, `lr_scheduler_type`, `warmup_ratio`.
- **LoRA**: `use_lora`, `lora_rank`, `lora_dropout`.
- **Stage**: `stage` controls which backbones are frozen (`lvp` = freeze LLM + vision + projector backbones, `lv` = freeze LLM + vision).

### 4.2 Launch DDP training

A typical 3-GPU launch (matching the comment at the top of [finetune_ddp.py](finetune_ddp.py#L1-L4)):

```bash
echo 1 | sudo tee /proc/sys/vm/drop_caches
NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=1 \
    torchrun --standalone --nnodes 1 --nproc-per-node 3 finetune_ddp.py
```

Adjust `--nproc-per-node` to match the number of GPUs on your machine, and make sure `global_batch_size` is divisible by `per_device_batch_size * num_gpus`.

Checkpoints are saved every `save_freq` steps under `<run_dir>/<run_id>/checkpoints/` as `step-XXXXXX-epoch-XX-loss=YYYY.pt`, plus a `latest-checkpoint.pt` symlink-style copy. To resume, set `resume=True`.

---

## 5. Evaluation on LIBERO

After fine-tuning, evaluate your checkpoint with [eval_libero.py](eval_libero.py).

### 5.1 Configure evaluation

`EvalConfig` (top of [eval_libero.py](eval_libero.py#L40-L54)) inherits from `RunConfig`, so most model-side fields are loaded automatically from the saved `run_config.jsonl`. You typically only need to set:

| Field | Meaning |
| --- | --- |
| `pretrained_vla_path` | Path to the same base OpenVLA-7B checkpoint used for training |
| `pretrained_checkpoint_path` | Directory of fine-tuned checkpoints (the `checkpoints/` folder under your `run_dir / run_id`) |
| `pretrained_checkpoint_step` | The training step of the checkpoint to evaluate |
| `task_suite_name` | One of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `device` | GPU index |
| `num_trials_per_task` | Number of rollouts per task (default `30`) |

### 5.2 Run evaluation

```bash
python eval_libero.py \
    --pretrained_checkpoint_path /path/to/run_dir/<run_id>/checkpoints/ \
    --pretrained_checkpoint_step 30000 \
    --task_suite_name libero_goal \
    --device 0
```

Rendering uses headless MuJoCo via OSMesa (`MUJOCO_GL=osmesa`). On a server with a GPU but no display you may instead use `egl`; on a workstation with a display, `glfw` works too — edit the line at the top of [eval_libero.py](eval_libero.py#L7).

Per-task success rates and rollout videos are written under `<run_dir>/<run_id>/` and (if a wandb project is configured) logged to Weights & Biases.
