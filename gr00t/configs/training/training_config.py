# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Output
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # Basic training
    max_steps: int = 30000  # this will override num_epochs
    global_batch_size: int = 1024
    batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # this will override warmup_ratio
    max_grad_norm: float = 1.0

    # Optimizer choice (huggingface TrainingArguments.optim)
    # Options include: 'adamw_torch', 'adamw_torch_fused', 'paged_adamw_32bit',
    # 'paged_adamw_8bit' (requires bitsandbytes), 'adafactor', etc.
    optim: str = "adamw_torch_fused"

    start_from_checkpoint: Optional[str] = None
    skip_weight_loading: bool = False  # skip loading checkpoint weights (architecture only)

    # Mixed precision
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 5

    # Model saving
    save_vl_model: bool = False  # Control whether to save VL model and processor in callbacks
    save_only_model: bool = False  # Skip optimizer/scheduler/RNG states — cannot resume training

    # Checkpoint uploading
    upload_checkpoints: bool = False
    upload_every: int = 1000
    upload_last_n_checkpoints: int = 5
    max_concurrent_uploads: int = 2

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: int = 500
    eval_set_split_ratio: float = 0.1
    eval_batch_size: int = 2
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_better: bool = True

    # DeepSpeed (default)
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3)
    gradient_checkpointing: bool = False

    # Transformers loading parameters
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = False
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  # Access token for HuggingFace Hub

    # DDP
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100

    # Hardware
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    # Data handling
    remove_unused_columns: bool = False

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"

    # Profiling
    enable_profiling: bool = False

    # Max number of retries in training for fault tolerance
    max_retries: int = 3

    # For testing.
    assert_loss_less_than: float | None = None

    # -----------------------------------------------------------------------
    # RL reward-informed training
    # -----------------------------------------------------------------------

    add_rl_callback: bool = False
    """Enable reward-informed loss scaling via sim rollouts.

    When True, RLCallback runs `rl_n_rollout_episodes` episodes every
    `rl_rollout_interval` training steps, computes shaped rewards, and
    uses the running-mean reward to modulate the imitation loss weight
    (see Gr00tTrainer.compute_loss in trainer.py).

    Requires a valid `rl_env_name` and a registered sim environment.
    """

    rl_env_name: str = ""
    """Gymnasium environment id for rollouts.

    Examples:
        ``"libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"``
        ``"simpler_env_google/google_robot_pick_object"``

    Must be non-empty when add_rl_callback=True.
    """

    rl_task_type: str = "manipulation"
    """Reward function family.  Either ``"manipulation"`` or ``"locomotion"``.

    - ``"manipulation"``  →  combined_manipulation_reward() (palm proximity + grasp + success)
    - ``"locomotion"``    →  foot_placement_reward() (stable stride width)
    """

    rl_object_name: str = "apple"
    """MuJoCo body name of the target object for manipulation reward.

    Used by ``_get_object_pos`` in RLCallback to read the object's xyz
    from the simulator.  Adjust to match the body name in your task's
    BDDL / XML (e.g. ``"cube_main"``, ``"bowl_1"``).
    """

    rl_rollout_interval: int = 100
    """Run sim rollouts every N training steps.

    Lower values give faster reward feedback but slow down training.
    Recommended range: 50–500 depending on episode length and GPU budget.
    """

    rl_n_rollout_episodes: int = 4
    """Number of rollout episodes collected per rollout call.

    More episodes → lower variance reward estimate, higher wall-clock cost.
    """

    rl_weight: float = 0.05
    """Strength of the reward signal on the imitation loss.

    The imitation loss is scaled by ``(1 + rl_weight * (1 - reward_ema))``:
    - When reward_ema ≈ 0 (poor policy): loss × (1 + rl_weight)  → train harder
    - When reward_ema ≈ 1 (good policy): loss × 1.0              → normal training

    Keep small (0.01–0.10) to avoid destabilising the imitation baseline.
    """

    # -----------------------------------------------------------------------
    # Open-loop evaluation
    # -----------------------------------------------------------------------

    # Open-loop evaluation
    enable_open_loop_eval: bool = False
    """Enable open-loop evaluation on saved checkpoints."""

    open_loop_eval_traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    open_loop_eval_steps_per_traj: int = 100
    """Number of steps to evaluate per trajectory."""

    open_loop_eval_plot_indices: Optional[list[int]] = None
    """List of action indices to plot. If None, plots all indices."""
