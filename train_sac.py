import argparse
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment_setup import make_env


class SafetyToGymnasiumWrapper(gym.Wrapper):
    """
    Adapts Safety-Gymnasium's 6-tuple step output to the 5-tuple API expected by SB3.

    Original:
        obs, reward, cost, terminated, truncated, info

    Wrapped:
        obs, reward, terminated, truncated, info

    The cost is preserved in info["cost"] and cumulative episode cost is tracked
    in info["episode_cost"]. This does NOT modify the environment's reward/cost.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        self.episode_cost += float(cost)
        self.episode_reward += float(reward)
        self.episode_length += 1

        info = dict(info)
        info["cost"] = float(cost)
        info["episode_cost"] = self.episode_cost
        info["episode_reward"] = self.episode_reward
        info["episode_length"] = self.episode_length

        if terminated or truncated:
            info.setdefault("episode", {})
            info["episode"]["r"] = self.episode_reward
            info["episode"]["l"] = self.episode_length
            info["episode"]["c"] = self.episode_cost

        return obs, reward, terminated, truncated, info


class CostLoggingCallback(BaseCallback):
    """Logs episode cost statistics from env infos during SB3 training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_costs: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                if "c" in ep:
                    self.episode_costs.append(float(ep["c"]))
                    self.logger.record("rollout/ep_cost", float(ep["c"]))
                if "r" in ep:
                    self.episode_rewards.append(float(ep["r"]))
                if "l" in ep:
                    self.episode_lengths.append(int(ep["l"]))

        if self.episode_costs:
            self.logger.record(
                "rollout/ep_cost_mean_100",
                float(np.mean(self.episode_costs[-100:])),
            )
        return True


def build_env(
    seed: int,
    normalize_obs: bool,
    smooth_actions: bool,
    action_alpha: float,
    frame_stack: int,
    update_obs_stats: bool,
    render_mode: Optional[str] = None,
):
    env = make_env(
        seed=seed,
        normalize_obs=normalize_obs,
        smooth_actions=smooth_actions,
        action_alpha=action_alpha,
        frame_stack=frame_stack,
        update_obs_stats=update_obs_stats,
        render_mode=render_mode,
    )
    env = SafetyToGymnasiumWrapper(env)
    env = Monitor(env)
    return env


def make_vec_env(
    seed: int,
    normalize_obs: bool,
    smooth_actions: bool,
    action_alpha: float,
    frame_stack: int,
    update_obs_stats: bool,
):
    return DummyVecEnv([
        lambda: build_env(
            seed=seed,
            normalize_obs=normalize_obs,
            smooth_actions=smooth_actions,
            action_alpha=action_alpha,
            frame_stack=frame_stack,
            update_obs_stats=update_obs_stats,
            render_mode=None,
        )
    ])


class FrozenStatsEvalCallback(EvalCallback):
    """
    Eval callback that optionally copies obs-normalization stats from train env to eval env.
    """

    def _on_step(self) -> bool:
        return super()._on_step()


def main():
    parser = argparse.ArgumentParser(description="Train SAC baseline on SafetyRacecarButton2-v0")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs/sac_baseline")
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--smooth-actions", action="store_true")
    parser.add_argument("--action-alpha", type=float, default=0.8)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    train_env = make_vec_env(
        seed=args.seed,
        normalize_obs=args.normalize_obs,
        smooth_actions=args.smooth_actions,
        action_alpha=args.action_alpha,
        frame_stack=args.frame_stack,
        update_obs_stats=True,
    )

    eval_env = make_vec_env(
        seed=5000,
        normalize_obs=args.normalize_obs,
        smooth_actions=False,
        action_alpha=args.action_alpha,
        frame_stack=args.frame_stack,
        update_obs_stats=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
    )

    callbacks = CallbackList([
        CostLoggingCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=args.log_dir,
            log_path=args.log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
    ])

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(args.log_dir, "final_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()