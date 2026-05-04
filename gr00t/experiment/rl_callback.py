# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
RL Callback for GR00T training.

Architecture
------------
ReplayBuffer
    Lightweight ring-buffer that stores per-step reward scalars and
    episode metadata.  It does NOT store full observations/images — that
    would require tens of GB.  The reward signal is surfaced to
    Gr00tTrainer via `replay_buffer.mean_reward()`, which the trainer
    uses to scale its imitation loss (see trainer.py).

RLCallback  (TrainerCallback)
    Runs sim rollouts every `rollout_interval` training steps.
    Uses the current model weights (captured in on_train_begin) so
    rollouts always reflect the latest policy.

    Per-step reward uses the functions in reward.py:
      - "manipulation" task  →  combined_manipulation_reward()
      - "locomotion"   task  →  foot_placement_reward()

Notes
-----
- Only runs on rank-0 to avoid duplicate rollouts in multi-GPU runs.
- The model is set to eval() during rollouts and restored to train() afterwards.
- For LIBERO envs: object positions are read from MuJoCo sim internals.
- DDP / DeepSpeed wrappers are unwrapped via _unwrap_model().
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback

from gr00t.experiment.reward import combined_manipulation_reward, foot_placement_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Ring-buffer that keeps track of per-step rewards from sim rollouts.

    Gr00tTrainer reads `mean_reward()` from this buffer every training step
    and uses it to modulate the imitation loss (see trainer.py).
    """

    def __init__(self, max_size: int = 5000):
        self._rewards: deque[float] = deque(maxlen=max_size)
        self._episodes: deque[dict] = deque(maxlen=200)   # episode-level summaries

    # ------------------------------------------------------------------
    def push_episode(self, step_rewards: list[float], meta: dict) -> None:
        """Store all per-step rewards from one episode."""
        for r in step_rewards:
            self._rewards.append(float(r))
        self._episodes.append({**meta, "mean_reward": float(np.mean(step_rewards)) if step_rewards else 0.0})

    def mean_reward(self) -> float:
        """Running mean over all stored rewards (up to max_size steps)."""
        if not self._rewards:
            return 0.0
        return float(np.mean(list(self._rewards)))

    def last_episode_mean(self) -> float:
        if not self._episodes:
            return 0.0
        return float(self._episodes[-1]["mean_reward"])

    def __len__(self) -> int:
        return len(self._rewards)


# ---------------------------------------------------------------------------
# RL Callback
# ---------------------------------------------------------------------------

class RLCallback(TrainerCallback):
    """
    Runs periodic sim rollouts during GR00T training and feeds shaped
    reward information back to Gr00tTrainer via a shared ReplayBuffer.

    Parameters
    ----------
    env_name : str
        Gymnasium environment id, e.g.
        ``"libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"``
    modality_configs : dict
        Modality config dict for the active embodiment, e.g.
        ``config.data.modality_configs["libero_sim"]``
    processor : Gr00tN1d7Processor
        The *same* processor used for training (already has statistics set).
    replay_buffer : ReplayBuffer
        Shared with Gr00tTrainer; written here, read there.
    task_type : str
        ``"manipulation"`` or ``"locomotion"``
    object_name : str
        MuJoCo body name of the target object, e.g. ``"apple"``.
        Only used for manipulation tasks.
    rollout_interval : int
        Run rollouts every N training steps.
    n_rollout_episodes : int
        Number of episodes to run per rollout call.
    action_horizon : int
        How many action steps to execute between policy predictions
        (open-loop chunk execution).
    embodiment_tag : str
        Embodiment tag string, e.g. ``"libero_sim"``.
    """

    def __init__(
        self,
        env_name: str,
        modality_configs: dict,
        processor: Any,
        replay_buffer: ReplayBuffer,
        task_type: str = "manipulation",
        object_name: str = "apple",
        rollout_interval: int = 100,
        n_rollout_episodes: int = 4,
        action_horizon: int = 8,
        embodiment_tag: str = "libero_sim",
    ) -> None:
        self.env_name           = env_name
        self.modality_configs   = modality_configs
        self.processor          = processor
        self.replay_buffer      = replay_buffer
        self.task_type          = task_type
        self.object_name        = object_name
        self.rollout_interval   = rollout_interval
        self.n_rollout_episodes = n_rollout_episodes
        self.action_horizon     = action_horizon
        self.embodiment_tag     = embodiment_tag

        # Filled during on_train_begin so rollouts always use current weights
        self._model_ref: Any = None

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        """Capture a reference to the (possibly DDP/DeepSpeed-wrapped) model."""
        self._model_ref = model
        logger.info(
            "[RLCallback] Ready — env=%s task=%s rollout_every=%d steps",
            self.env_name, self.task_type, self.rollout_interval,
        )

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Run rollouts at the configured interval, rank-0 only."""
        # Only run on rank 0 to avoid duplicated sim instances
        if args.local_rank not in (-1, 0):
            return
        if state.global_step == 0:
            return
        if state.global_step % self.rollout_interval != 0:
            return
        if self._model_ref is None:
            logger.warning("[RLCallback] No model reference — skipping rollout.")
            return

        logger.info(
            "[RLCallback] Step %d — running %d rollout episode(s)…",
            state.global_step, self.n_rollout_episodes,
        )

        model = self._model_ref
        underlying = self._unwrap_model(model)
        underlying.eval()
        self.processor.eval()

        all_rewards: list[float] = []
        for ep_idx in range(self.n_rollout_episodes):
            try:
                episode_rewards = self._run_episode(underlying)
            except Exception as exc:
                logger.warning("[RLCallback] Episode %d failed: %s", ep_idx, exc)
                episode_rewards = [0.0]

            self.replay_buffer.push_episode(
                episode_rewards,
                meta={"global_step": state.global_step, "ep_idx": ep_idx},
            )
            all_rewards.extend(episode_rewards)

        underlying.train()
        self.processor.train()

        if all_rewards:
            logger.info(
                "[RLCallback] Episode mean reward: %.4f  |  Buffer mean: %.4f",
                float(np.mean(all_rewards)),
                self.replay_buffer.mean_reward(),
            )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _run_episode(self, model) -> list[float]:
        """
        Run one full episode in the sim environment.

        Returns a list of per-step shaped rewards.
        """
        import gymnasium as gym

        self._register_env()
        env = gym.make(self.env_name)
        obs, _info = env.reset()
        done = False
        step_rewards: list[float] = []
        step_count = 0
        max_steps = 500   # safety cap

        while not done and step_count < max_steps:
            # Format obs → model inputs (handles batch/temporal dims)
            try:
                model_inputs = self._format_obs(obs)
            except Exception as exc:
                logger.warning("[RLCallback] _format_obs failed: %s", exc)
                break

            # Get action chunk from current policy
            with torch.no_grad():
                action_pred = model.get_action(model_inputs)

            # Decode full action chunk to env-compatible format
            decoded = self._decode_action_chunk(action_pred, obs)

            # Execute chunk open-loop
            for t in range(self.action_horizon):
                if done or step_count >= max_steps:
                    break

                step_action = {k: decoded[k][t] for k in decoded}

                # Object position for reward (manipulation)
                object_pos = self._get_object_pos(env)

                next_obs, _rew, terminated, truncated, info = env.step(step_action)
                done = terminated or truncated

                reward = self._compute_reward(obs, next_obs, info, object_pos)
                step_rewards.append(reward)

                obs = next_obs
                step_count += 1

        env.close()
        return step_rewards

    # ------------------------------------------------------------------
    # Environment registration helpers
    # ------------------------------------------------------------------

    def _register_env(self) -> None:
        if "libero" in self.env_name:
            from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
            register_libero_envs()
        elif "simpler_env_google" in self.env_name or "simpler_env_widowx" in self.env_name:
            from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
            register_simpler_envs()

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _format_obs(self, obs: dict) -> Any:
        """
        Convert a flat gym env observation dict to model-ready inputs by
        calling ``processor.process_observation``.

        Adds the required (B=1, T=1) batch and temporal dimensions that
        the processor expects.

        Currently implemented for LIBERO.  Extend the elif chain for
        other environments.
        """
        from gr00t.data.embodiment_tags import EmbodimentTag

        emb_tag = EmbodimentTag(self.embodiment_tag)

        if "libero" in self.env_name:
            return self._format_libero_obs(obs, emb_tag)

        # -- add further environments here --
        raise NotImplementedError(
            f"Observation formatting not implemented for env '{self.env_name}'. "
            f"Add a branch in RLCallback._format_obs."
        )

    def _format_libero_obs(self, obs: dict, emb_tag) -> Any:
        """
        LIBERO specific formatting.

        Raw LIBERO obs keys (from LiberoEnv._process_observation):
            video.image          : (H, W, 3) uint8
            video.wrist_image    : (H, W, 3) uint8
            state.x/y/z/…        : list[float] or ndarray
            annotation.human.action.task_description : str

        We add B=1, T=1 dimensions and call processor.process_observation.
        """
        nested: dict[str, Any] = {}

        # Video — (H, W, C) → (B=1, T=1, H, W, C)
        for view_key in self.modality_configs["video"].modality_keys:
            raw = np.asarray(obs[f"video.{view_key}"], dtype=np.uint8)
            nested[f"video.{view_key}"] = raw[np.newaxis, np.newaxis]   # (1,1,H,W,C)

        # State — scalar/array → (B=1, T=1, D)
        for state_key in self.modality_configs["state"].modality_keys:
            raw = np.asarray(obs[f"state.{state_key}"], dtype=np.float32).flatten()
            nested[f"state.{state_key}"] = raw[np.newaxis, np.newaxis]  # (1,1,D)

        # Language — str → list[str] of length B=1
        lang_key = self.modality_configs["language"].modality_keys[0]
        nested[lang_key] = [obs[lang_key]]

        return self.processor.process_observation(nested, emb_tag)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action_chunk(self, action_pred: dict, obs: dict) -> dict[str, np.ndarray]:
        """
        Decode the model's normalised action prediction into env-compatible
        arrays for each of the ``action_horizon`` steps.

        Returns a dict {env_action_key: ndarray(action_horizon, D)}.
        """
        from gr00t.data.embodiment_tags import EmbodimentTag

        emb_tag = EmbodimentTag(self.embodiment_tag)

        # action_pred["action_pred"] shape: (B=1, max_action_horizon, max_action_dim)
        normalized = action_pred["action_pred"].cpu().numpy()   # (1, T, D)

        # Build state dict for relative-action decoding (LIBERO uses absolute,
        # but keep this general for other embodiments).
        state_keys = self.modality_configs["state"].modality_keys
        batched_states: dict[str, np.ndarray] = {}
        for key in state_keys:
            raw = np.asarray(obs[f"state.{key}"], dtype=np.float32).flatten()
            batched_states[key] = raw[np.newaxis, np.newaxis]  # (1, 1, D)

        # decode_action returns {joint_key: (B, T, D)} in physical units
        decoded = self.processor.decode_action(normalized, emb_tag, state=batched_states)

        # Build {env_action_key: (action_horizon, D)} — squeeze batch dim, take first H steps
        result: dict[str, np.ndarray] = {}
        action_keys = self.modality_configs["action"].modality_keys
        for key in action_keys:
            arr = decoded[key][0]              # (T, D)  — drop batch dim
            result[f"action.{key}"] = arr[:self.action_horizon]
        return result

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self, obs: dict, next_obs: dict, info: dict, object_pos: np.ndarray
    ) -> float:
        if self.task_type == "manipulation":
            palm_pos = np.array([
                float(np.asarray(obs["state.x"]).flat[0]),
                float(np.asarray(obs["state.y"]).flat[0]),
                float(np.asarray(obs["state.z"]).flat[0]),
            ], dtype=np.float32)
            # gripper: last element of gripper state (1 = closed)
            gripper_raw = np.asarray(obs.get("state.gripper", [0.0]), dtype=np.float32).flat[-1]
            success = bool(info.get("success", False))
            return combined_manipulation_reward(palm_pos, object_pos, float(gripper_raw), success)

        elif self.task_type == "locomotion":
            # For Unitree G1: state.left_leg / state.right_leg contain joint angles.
            # Use the last 3 values of each as a positional proxy.
            left  = np.asarray(obs.get("state.left_leg",  np.zeros(6)), dtype=np.float32)
            right = np.asarray(obs.get("state.right_leg", np.zeros(6)), dtype=np.float32)
            return foot_placement_reward(left[-3:], right[-3:])

        else:
            raise ValueError(f"Unknown task_type: '{self.task_type}'. Use 'manipulation' or 'locomotion'.")

    def _get_object_pos(self, env) -> np.ndarray:
        """
        Extract the target object's xyz position from the MuJoCo simulator.

        For LIBERO, this accesses ``env.unwrapped._env.sim`` directly.
        Returns zeros if the lookup fails (reward will be near-zero that step).
        """
        try:
            if "libero" in self.env_name:
                sim = env.unwrapped._env.sim
                body_id = sim.model.body_name2id(self.object_name)
                return sim.data.body_xpos[body_id].copy().astype(np.float32)
        except Exception as exc:
            logger.debug("[RLCallback] _get_object_pos failed (%s), returning zeros.", exc)
        return np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        """
        Unwrap DDP or DeepSpeed wrappers to reach the underlying Gr00tN1d7.
        Gr00tN1d7.get_action is not forwarded by DDP's __getattr__, so we
        must access module directly.
        """
        if hasattr(model, "module"):   # torch.nn.parallel.DistributedDataParallel
            return model.module
        return model
