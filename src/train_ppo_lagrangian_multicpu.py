import argparse
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment_setup import make_env


class LagrangianWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that penalizes safety costs dynamically.
    Replaces reward with:
        shaped_reward = reward - beta * cost
    where beta is a dynamically adjusted Lagrangian multiplier.
    """

    def __init__(self, env: gym.Env, beta_holder: List[float]):
        super().__init__(env)
        self.beta_holder = beta_holder

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        beta = self.beta_holder[0]
        shaped_reward = reward - beta * cost
        return obs, shaped_reward, cost, terminated, truncated, info


class SafetyToGymnasiumWrapper(gym.Wrapper):
    """
    Adapts Safety-Gymnasium's 6-tuple step output to the 5-tuple API expected by SB3.
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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                cost = float(info.get("episode_cost", 0.0))
                self.episode_costs.append(cost)
                self.logger.record("rollout/ep_cost", cost)

        if self.episode_costs:
            self.logger.record(
                "rollout/ep_cost_mean_100",
                float(np.mean(self.episode_costs[-100:])),
            )
        return True


class LagrangianCallback(BaseCallback):
    """
    Adjusts the Lagrangian multiplier (beta) based on policy performance.
    gradient ascent: beta = beta + lr * (average_cost - cost_limit)
    """

    def __init__(
        self,
        beta_holder: List[float],
        cost_limit: float,
        lr: float,
        max_beta: float,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.beta_holder = beta_holder
        self.cost_limit = cost_limit
        self.lr = lr
        self.max_beta = max_beta
        self.episode_costs: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                cost = float(info.get("episode_cost", 0.0))
                self.episode_costs.append(cost)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_costs) > 0:
            avg_cost = np.mean(self.episode_costs)
            # Dynamic multiplier update with clamp
            new_beta = self.beta_holder[0] + self.lr * (avg_cost - self.cost_limit)
            self.beta_holder[0] = min(max(0.0, new_beta), self.max_beta)
            self.logger.record("train/lagrangian_beta", self.beta_holder[0])
            self.logger.record("train/rollout_cost_mean", avg_cost)
            print(
                f"\n[Lagrangian] Rollout ended. Avg Cost: {avg_cost:.2f} "
                f"(Limit: {self.cost_limit}), New Beta: {self.beta_holder[0]:.5f}"
            )
            self.episode_costs.clear()


class StatsSyncEvalCallback(EvalCallback):
    """
    EvalCallback that synchronizes observation normalization stats
    from the multi-process training environment to the evaluation environment.
    """
    def __init__(self, train_env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            try:
                # Use get_attr to safely pull the 'rms' object across CPU processes
                train_rms = self.train_env.get_attr("rms", indices=[0])[0]

                # Find ObsNormWrapper in eval_env (Since eval_env is DummyVecEnv, envs[0] works)
                eval_wrapper = None
                current_env = self.eval_env.envs[0]
                
                while hasattr(current_env, "env"):
                    if type(current_env).__name__ == "ObsNormWrapper":
                        eval_wrapper = current_env
                        break
                    current_env = current_env.env
                    
                if type(current_env).__name__ == "ObsNormWrapper":
                    eval_wrapper = current_env

                # Apply the stats
                if eval_wrapper is not None and train_rms is not None:
                    eval_wrapper.rms.mean = train_rms.mean.copy()
                    eval_wrapper.rms.var = train_rms.var.copy()
                    eval_wrapper.rms.count = train_rms.count
                    if self.verbose > 0:
                        print("[Evaluation Stats Sync] Synchronized normalization stats to eval_env.")
            except Exception as e:
                print(f"[Warning] Failed to synchronize normalizer stats: {e}")
        return super()._on_step()


def build_env(
    seed: int,
    normalize_obs: bool,
    smooth_actions: bool,
    action_alpha: float,
    frame_stack: int,
    update_obs_stats: bool,
    beta_holder: List[float],
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
    env = LagrangianWrapper(env, beta_holder)
    env = SafetyToGymnasiumWrapper(env)
    env = Monitor(env)
    return env


def make_vec_env(
    seed: int,
    n_envs: int,
    normalize_obs: bool,
    smooth_actions: bool,
    action_alpha: float,
    frame_stack: int,
    update_obs_stats: bool,
    beta_holder: List[float],
):
    def make_env_fn(rank):
        return lambda: build_env(
            seed=seed + rank,
            normalize_obs=normalize_obs,
            smooth_actions=smooth_actions,
            action_alpha=action_alpha,
            frame_stack=frame_stack,
            update_obs_stats=update_obs_stats,
            beta_holder=beta_holder,
            render_mode=None,
        )
    
    env_fns = [make_env_fn(i) for i in range(n_envs)]
    
    # Run SubprocVecEnv for training (Multi-Core), DummyVecEnv for eval (Single-Core)
    return SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)


def main():
    parser = argparse.ArgumentParser(description="Train PPO-Lagrangian Multi-CPU")
    parser.add_argument("--total-timesteps", type=int, default= 1800000)
    parser.add_argument("--seed", type=int, default=0)
    
    # Resolve the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "src" else script_dir
    default_log_dir = os.path.join(project_root, "runs", "ppo_true_lag_perfect_scale")
    
    parser.add_argument("--log-dir", type=str, default=default_log_dir)
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel CPU processes")
    parser.add_argument("--normalize-obs", action="store_true", default=True)
    parser.add_argument("--smooth-actions", action="store_true", default=True)
    parser.add_argument("--action-alpha", type=float, default=0.8)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost-limit", type=float, default=25.0)
    
    parser.add_argument("--lagrangian-lr", type=float, default=0.00001)
    parser.add_argument("--max-beta", type=float, default=0.015)
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Curiosity engine")
    
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    beta_holder = [0.0]

    train_env = make_vec_env(
        seed=args.seed,
        n_envs=args.n_envs,
        normalize_obs=args.normalize_obs,
        smooth_actions=args.smooth_actions,
        action_alpha=args.action_alpha,
        frame_stack=args.frame_stack,
        update_obs_stats=True,
        beta_holder=beta_holder,
    )

    eval_env = make_vec_env(
        seed=5000,
        n_envs=1, # Keep evaluation strictly to 1 car
        normalize_obs=args.normalize_obs,
        smooth_actions=args.smooth_actions,
        action_alpha=args.action_alpha,
        frame_stack=args.frame_stack,
        update_obs_stats=False,
        beta_holder=beta_holder, 
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
        ent_coef=args.ent_coef,
    )

    # Note: Using max(1, ...) so it scales correctly with multiple environments
    eval_freq = max(1, args.eval_freq // args.n_envs)

    callbacks = CallbackList([
        CostLoggingCallback(),
        LagrangianCallback(
            beta_holder=beta_holder,
            cost_limit=args.cost_limit,
            lr=args.lagrangian_lr,
            max_beta=args.max_beta
        ),
        StatsSyncEvalCallback(
            train_env=train_env,
            eval_env=eval_env,
            best_model_save_path=args.log_dir,
            log_path=args.log_dir,
            eval_freq=eval_freq,
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

    if args.normalize_obs:
        try:
            stats_path = os.path.join(args.log_dir, "obs_stats.npz")
            # Ask the multi-core engine to safely save the stats from process 0
            train_env.env_method("save_stats", stats_path, indices=[0])
            print(f"Saved observation normalization stats to {stats_path}")
        except Exception as e:
            print(f"Failed to save observation stats: {e}")

if __name__ == "__main__":
    main()