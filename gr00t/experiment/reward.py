# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Reward functions for RL-informed training of GR00T policies.

These are pure functions — no sim dependencies — so they are easy to unit-test
independently of any environment.

Two task families are supported:
  - Manipulation  (e.g. "grab the apple"):  palm-proximity + grasp + success bonus
  - Locomotion    (e.g. walking gait):       foot-placement stability reward

All rewards are normalised to roughly [0, 1] so that the RL weight in
TrainingConfig.rl_weight is easy to reason about.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Manipulation helpers
# ---------------------------------------------------------------------------

def palm_proximity_reward(
    palm_pos: np.ndarray,
    object_pos: np.ndarray,
) -> float:
    """
    Dense reward for end-effector / palm proximity to a target object.

    Uses an exponential decay so the gradient is always non-zero and
    the reward peaks at 1.0 when the palm is directly at the object.

    Args:
        palm_pos:   (3,) xyz of end-effector / palm in world frame (metres)
        object_pos: (3,) xyz of target object (e.g. apple) in world frame

    Returns:
        float in (0, 1]
    """
    distance = float(np.linalg.norm(
        np.asarray(palm_pos, dtype=np.float64)
        - np.asarray(object_pos, dtype=np.float64)
    ))
    # decay constant 10 → reward ≈ 0.37 at 0.1 m, ≈ 0.02 at 0.4 m
    return float(np.exp(-10.0 * distance))


def grasp_bonus(
    palm_pos: np.ndarray,
    object_pos: np.ndarray,
    gripper_state: float,
    close_threshold: float = 0.05,
) -> float:
    """
    Binary bonus awarded when the gripper is closed while the palm is near
    the object.  Encourages the policy to actually close its hand rather than
    just hover next to the target.

    Args:
        palm_pos:        (3,) xyz of palm
        object_pos:      (3,) xyz of target object
        gripper_state:   scalar in [0, 1] where 0 = fully open, 1 = fully closed
        close_threshold: metres — must be within this distance to earn the bonus

    Returns:
        1.0 if gripper is closed and palm is close, else 0.0
    """
    distance = float(np.linalg.norm(
        np.asarray(palm_pos, dtype=np.float64)
        - np.asarray(object_pos, dtype=np.float64)
    ))
    if gripper_state > 0.5 and distance < close_threshold:
        return 1.0
    return 0.0


def combined_manipulation_reward(
    palm_pos: np.ndarray,
    object_pos: np.ndarray,
    gripper_state: float,
    success: bool,
) -> float:
    """
    Combined reward for a pick-and-place style manipulation task.

    Breakdown (raw scale):
      - Proximity reward:  [0, 1]   dense signal guiding palm toward object
      - Grasp bonus:       {0, 1}   binary, triggers when gripper closes on object
      - Success bonus:     {0, 5}   large terminal reward for task completion

    The raw range is [0, 7].  We normalise to [0, 1] by dividing by 7 so
    that rl_weight has consistent meaning across task types.

    Args:
        palm_pos:      (3,) end-effector xyz
        object_pos:    (3,) target object xyz
        gripper_state: scalar in [0, 1], 1 = closed
        success:       True if the episode was solved this step

    Returns:
        float in [0, 1]
    """
    proximity = palm_proximity_reward(palm_pos, object_pos)
    grasp     = grasp_bonus(palm_pos, object_pos, gripper_state)
    terminal  = 5.0 if success else 0.0

    raw = proximity + grasp + terminal
    return float(np.clip(raw / 7.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Locomotion helpers
# ---------------------------------------------------------------------------

def foot_placement_reward(
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    min_stride: float = 0.10,
    max_stride: float = 0.40,
) -> float:
    """
    Reward for a stable, natural walking gait.

    Penalises:
      - Feet too far apart  → unstable, likely to topple
      - Feet too close      → shuffling, inefficient gait

    Uses a Gaussian bell centred on the ideal stride
    (midpoint of [min_stride, max_stride]).

    Args:
        left_foot_pos:  (3,) position of left foot (or last-3 joint angles as proxy)
        right_foot_pos: (3,) position of right foot
        min_stride:     minimum acceptable inter-foot distance (m)
        max_stride:     maximum acceptable inter-foot distance (m)

    Returns:
        float in [0, 1]
    """
    left  = np.asarray(left_foot_pos,  dtype=np.float64)
    right = np.asarray(right_foot_pos, dtype=np.float64)
    stride = float(np.linalg.norm(left - right))

    if stride < min_stride or stride > max_stride:
        return 0.0

    ideal = (min_stride + max_stride) / 2.0
    # σ = quarter of the valid range so the bell fits neatly
    sigma = (max_stride - min_stride) / 4.0
    return float(np.exp(-((stride - ideal) ** 2) / (2.0 * sigma ** 2)))
