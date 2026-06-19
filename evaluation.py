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
from controllers import ScriptedController
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
    obs_stats_path: Optional[str] = None,
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
    obs_stats_path : str or None
        If provided, loads observation normalization stats from this file.

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

    if normalize_obs and obs_stats_path:
        from environment_setup import ObsNormWrapper
        curr = env
        while hasattr(curr, "env") and not isinstance(curr, ObsNormWrapper):
            curr = curr.env
        if isinstance(curr, ObsNormWrapper):
            if os.path.exists(obs_stats_path):
                curr.load_stats(obs_stats_path)
                if verbose:
                    print(f"[Evaluation] Loaded observation normalization stats from {obs_stats_path}")
            else:
                print(f"[Warning] Normalization stats file not found at {obs_stats_path}. Using uninitialized stats.")

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
    print(f"  Evaluation Report - {controller_name}")
    print(sep)
    print(f"  Episodes            : {metrics.n_episodes}")
    print(f"  Mean reward         : {metrics.mean_reward:+.4f}  +/- {metrics.std_reward:.4f}")
    print(f"  [min, max] reward   : [{metrics.min_reward:+.4f}, {metrics.max_reward:+.4f}]")
    print(f"  Mean cost           : {metrics.mean_cost:.4f}  +/- {metrics.std_cost:.4f}")
    print(f"  Max cost (episode)  : {metrics.max_cost:.4f}")
    print(f"  Zero-cost episodes  : {metrics.zero_cost_rate*100:.1f}%")
    print(
        f"  Safe episodes       : {metrics.safe_episode_rate*100:.1f}%  "
        f"(cost <= {metrics.cost_threshold})"
    )
    print(f"  Mean episode length : {metrics.mean_episode_length:.1f} steps")
    print(f"  Combined score      : {metrics.combined_score:+.4f}  "
          f"(R - {metrics.lambda_cost} * C)")
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



def generate_and_save_stats(save_path: str, num_steps: int = 10000, controller = None):
    """
    Runs a policy (or random actions if None) for a few steps to collect observation normalization stats
    and saves them to the specified path. This is useful if training did not save the stats.
    """
    print(f"[Evaluation] Generating observation normalization stats using the policy over {num_steps} steps...")
    # Create the environment with update=True to accumulate stats
    env = make_env(normalize_obs=True, update_obs_stats=True)
    obs, info = env.reset(seed=42)
    if controller is not None:
        controller.reset(seed=42)
    
    for _ in range(num_steps):
        if controller is not None:
            action, _ = controller.act(obs)
        else:
            action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            if controller is not None:
                controller.reset()
            
    # Find ObsNormWrapper and save stats
    from environment_setup import ObsNormWrapper
    curr = env
    while hasattr(curr, "env") and not isinstance(curr, ObsNormWrapper):
        curr = curr.env
    
    if isinstance(curr, ObsNormWrapper):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        curr.save_stats(save_path)
        print(f"[Evaluation] Saved generated normalization stats to {save_path}")
    else:
        print("[Error] Could not find ObsNormWrapper in environment.")
    env.close()


if __name__ == "__main__":
    import argparse
    from controllers import SACController, ScriptedController

    parser = argparse.ArgumentParser(description="Evaluate a controller on SafetyRacecarButton2-v0")
    parser.add_argument("--controller", type=str, default="sac", choices=["sac", "scripted"])
    parser.add_argument("--model-path", type=str, default="runs/sac_baseline/best_model.zip")
    parser.add_argument("--obs-stats-path", type=str, default="runs/sac_baseline/obs_stats.npz")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Step 1: Load the selected controller
    if args.controller == "sac":
        if not os.path.exists(args.model_path):
            print(f"[Error] Model checkpoint not found at: {args.model_path}")
            exit(1)
        print(f"Loading SACController from {args.model_path}...")
        ctrl = SACController(args.model_path)
        controller_name = f"SAC ({os.path.basename(args.model_path)})"
        normalize_obs = True
    else:
        ctrl = ScriptedController()
        controller_name = "ScriptedController"
        normalize_obs = False

    # Step 2: For SAC, generate normalizer stats if they don't exist
    if normalize_obs and not os.path.exists(args.obs_stats_path):
        print(f"[Warning] Normalization stats not found at {args.obs_stats_path}")
        print("We will run the policy for 10,000 steps to automatically generate the stats file...")
        generate_and_save_stats(args.obs_stats_path, num_steps=10000, controller=ctrl)

    # Step 3: Run the evaluation
    print(f"Starting evaluation of {controller_name} over {args.n_episodes} episodes...")
    metrics = evaluate_policy(
        controller=ctrl,
        n_episodes=args.n_episodes,
        normalize_obs=normalize_obs,
        obs_stats_path=args.obs_stats_path if normalize_obs else None,
        verbose=True,
        render_mode=args.render,
    )
    print_report(metrics, controller_name=controller_name)


