"""
Project 1: Safe Mobile Robot Button Navigation
Official-style evaluation loop and metrics.

This module provides:
  - run_episode()       : runs one episode and returns per-step metrics
  - evaluate_policy()   : evaluates a controller over N episodes
  - compute_metrics()   : computes the task-specific scoring metrics
  - print_report()      : human-readable summary

Rules reminder
--------------
  Teams MUST NOT modify the reward function, cost function, termination
  logic, random seeds, or scoring script.  This file only READS reward and
  cost; it never touches the environment internals.
"""

import numpy as np
from controllers import RandomController
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import os

from environment_setup import make_env


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """All statistics collected for one evaluation episode."""
    seed: int
    total_reward: float
    total_cost: float
    episode_length: int
    # Per-step breakdown (optional, useful for analysis)
    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Aggregate metrics over all evaluation episodes."""
    n_episodes: int

    # Reward statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float

    # Cost statistics
    mean_cost: float
    std_cost: float
    max_cost: float

    # Safety statistics
    # fraction of episodes with zero cost violations
    zero_cost_rate: float
    # fraction of episodes that stay below the cost threshold
    safe_episode_rate: float

    # Robustness
    mean_episode_length: float

    # Combined score (higher is better):
    #   score = mean_reward - lambda_cost * mean_cost
    # lambda_cost penalizes unsafe behavior
    combined_score: float

    cost_threshold: float = 25.0
    lambda_cost: float = 1.0


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    controller,
    env,
    seed: int,
    max_steps: Optional[int] = None,
    record_steps: bool = False,
) -> EpisodeResult:
    """
    Runs one evaluation episode.

    Parameters
    ----------
    controller : Controller
        Must implement .reset(seed) and .act(observation) → (action, info).
    env : gymnasium.Env
        A wrapped SafetyRacecarButton2-v0 environment.
    seed : int
        Episode seed (used for env.reset and controller.reset).
    max_steps : int or None
        Hard cap on episode length (None = use env's own truncation).
    record_steps : bool
        Whether to store per-step reward/cost lists.

    Returns
    -------
    EpisodeResult
    """
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    rewards = []
    costs = []

    terminated = truncated = False

    while not (terminated or truncated):
        action, ctrl_info = controller.act(obs)

        # Safety-Gymnasium 6-tuple: obs, reward, cost, terminated, truncated, info
        obs, reward, cost, terminated, truncated, step_info = env.step(action)

        total_reward += reward
        total_cost += cost
        steps += 1

        if record_steps:
            rewards.append(float(reward))
            costs.append(float(cost))

        if max_steps is not None and steps >= max_steps:
            break

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        total_cost=total_cost,
        episode_length=steps,
        rewards=rewards if record_steps else [],
        costs=costs if record_steps else [],
    )


# ---------------------------------------------------------------------------
# Multi-episode evaluator
# ---------------------------------------------------------------------------

def evaluate_policy(
    controller,
    n_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    normalize_obs: bool = True,
    smooth_actions: bool = False,   # typically off during evaluation
    action_alpha: float = 0.8,
    frame_stack: int = 1,
    cost_threshold: float = 25.0,
    lambda_cost: float = 1.0,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    render_mode: bool = False,
) -> EvaluationMetrics:
    """
    Evaluates a controller across multiple randomized episodes.

    Parameters
    ----------
    controller : Controller
        Implements .reset(seed) and .act(observation) → (action, info).
    n_episodes : int
        Number of episodes to run.
    seeds : list of int or None
        Explicit seeds to use. If None, uses range(n_episodes).
    normalize_obs : bool
        Whether to apply observation normalization (frozen stats).
    smooth_actions : bool
        Whether to apply action smoothing during evaluation.
    action_alpha : float
        EMA coefficient for action smoothing.
    frame_stack : int
        Number of frames to stack.
    cost_threshold : float
        Maximum acceptable cumulative cost per episode for 'safe' rating.
    lambda_cost : float
        Weight of cost in the combined score.
    verbose : bool
        Whether to print per-episode results.
    save_dir : str or None
        If provided, saves per-episode results as JSON.

    Returns
    -------
    EvaluationMetrics
    """
    if seeds is None:
        seeds = list(range(n_episodes))

    env = make_env(
        normalize_obs=normalize_obs,
        smooth_actions=smooth_actions,
        action_alpha=action_alpha,
        frame_stack=frame_stack,
        update_obs_stats=False,     # freeze stats during evaluation
        render_mode=render_mode,
    )

    results: List[EpisodeResult] = []

    for i, seed in enumerate(seeds):
        result = run_episode(controller, env, seed=seed)
        results.append(result)

        if verbose:
            print(
                f"  Episode {i+1:>3d}/{n_episodes}  "
                f"seed={seed:<4d}  "
                f"R={result.total_reward:+8.3f}  "
                f"C={result.total_cost:7.3f}  "
                f"L={result.episode_length:>4d}"
            )

    env.close()
    metrics = compute_metrics(results, cost_threshold=cost_threshold, lambda_cost=lambda_cost)

    if save_dir is not None:
        _save_results(results, metrics, save_dir)

    return metrics


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[EpisodeResult],
    cost_threshold: float = 25.0,
    lambda_cost: float = 1.0,
) -> EvaluationMetrics:
    """
    Computes aggregate metrics from a list of EpisodeResult objects.

    Scoring
    -------
    combined_score = mean_reward - lambda_cost * mean_cost

    A higher combined score is better.  Controllers are ranked first by
    combined_score; in case of a tie, lower mean_cost wins.
    """
    rewards = np.array([r.total_reward for r in results])
    costs = np.array([r.total_cost for r in results])
    lengths = np.array([r.episode_length for r in results])

    zero_cost_rate = float(np.mean(costs == 0.0))
    safe_episode_rate = float(np.mean(costs <= cost_threshold))
    combined_score = float(np.mean(rewards)) - lambda_cost * float(np.mean(costs))

    return EvaluationMetrics(
        n_episodes=len(results),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        mean_cost=float(np.mean(costs)),
        std_cost=float(np.std(costs)),
        max_cost=float(np.max(costs)),
        zero_cost_rate=zero_cost_rate,
        safe_episode_rate=safe_episode_rate,
        mean_episode_length=float(np.mean(lengths)),
        combined_score=combined_score,
        cost_threshold=cost_threshold,
        lambda_cost=lambda_cost,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(metrics: EvaluationMetrics, controller_name: str = "Controller"):
    """Prints a formatted evaluation report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Evaluation Report — {controller_name}")
    print(sep)
    print(f"  Episodes            : {metrics.n_episodes}")
    print(f"  Mean reward         : {metrics.mean_reward:+.4f}  ± {metrics.std_reward:.4f}")
    print(f"  [min, max] reward   : [{metrics.min_reward:+.4f}, {metrics.max_reward:+.4f}]")
    print(f"  Mean cost           : {metrics.mean_cost:.4f}  ± {metrics.std_cost:.4f}")
    print(f"  Max cost (episode)  : {metrics.max_cost:.4f}")
    print(f"  Zero-cost episodes  : {metrics.zero_cost_rate*100:.1f}%")
    print(
        f"  Safe episodes       : {metrics.safe_episode_rate*100:.1f}%  "
        f"(cost ≤ {metrics.cost_threshold})"
    )
    print(f"  Mean episode length : {metrics.mean_episode_length:.1f} steps")
    print(f"  Combined score      : {metrics.combined_score:+.4f}  "
          f"(R - {metrics.lambda_cost}·C)")
    print(sep)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_controllers(
    results: Dict[str, EvaluationMetrics]
) -> None:
    """
    Prints a comparison table of multiple controller evaluations.

    Parameters
    ----------
    results : dict
        Mapping of controller name → EvaluationMetrics.
    """
    header = (
        f"{'Controller':<30}  {'Mean R':>8}  {'Mean C':>8}  "
        f"{'Safe%':>6}  {'Score':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"  {name:<28}  "
            f"{m.mean_reward:>+8.3f}  "
            f"{m.mean_cost:>8.3f}  "
            f"{m.safe_episode_rate*100:>5.1f}%  "
            f"{m.combined_score:>+8.3f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_results(
    results: List[EpisodeResult],
    metrics: EvaluationMetrics,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    episodes_path = os.path.join(save_dir, "episodes.json")
    metrics_path = os.path.join(save_dir, "metrics.json")

    episodes_data = [
        {
            "seed": r.seed,
            "total_reward": r.total_reward,
            "total_cost": r.total_cost,
            "episode_length": r.episode_length,
        }
        for r in results
    ]

    with open(episodes_path, "w") as f:
        json.dump(episodes_data, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"[Evaluation] Results saved to {save_dir}")



if __name__ == "__main__":

    env_tmp = make_env(normalize_obs=False, smooth_actions=False, render_mode=False)
    ctrl = RandomController(env_tmp.action_space)
    env_tmp.close()

    metrics = evaluate_policy(
        controller=ctrl,
        n_episodes=3,
        verbose=True,
        render_mode=True,
    )
    print_report(metrics, controller_name="RandomController")
