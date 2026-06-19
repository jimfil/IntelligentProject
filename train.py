"""
Project 1: Safe Mobile Robot Button Navigation
Training scaffold — wires together the environment, a controller,
and the logging/evaluation infrastructure.

HOW TO USE
----------
1. Implement your controller in controllers.py (not provided here).
2. Import it below and pass it to the Trainer.
3. Run:  python train.py --controller my_method --seed 0

The scaffold handles:
  - Environment creation and seeding
  - Observation normalization stat collection
  - Episode logging (CSV + TensorBoard)
  - Periodic evaluation with frozen obs stats
  - Checkpoint saving (controller parameters)
  - Final evaluation report

Rules reminder
--------------
  Do NOT modify the benchmark environment, reward, cost, termination logic,
  random seeds for official evaluation, or the scoring script.
  You MAY use observation normalization, action smoothing, frame stacking,
  recurrent state, memory buffers, learned representations, demonstrations,
  or model-based planning — as long as the final controller uses only the
  observations allowed by the benchmark.
"""

import argparse
import os
import time

from environment_setup import make_env
from evaluation import evaluate_policy, print_report, compare_controllers
from utils import EpisodeLogger, ScalarWriter, AverageMeter, set_global_seed, print_stats

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training loop scaffold.

    This class is intentionally controller-agnostic.  Any controller that
    exposes the benchmark interface:

        controller.reset(seed=None)
        action, info = controller.act(observation)

    can be dropped in.  For learning-based controllers that need to be
    updated, add a controller.update(batch) call inside the loop.
    """

    def __init__(
        self,
        controller,
        controller_name: str = "my_controller",
        log_dir: str = "runs",
        # Environment settings
        normalize_obs: bool = True,
        smooth_actions: bool = True,
        action_alpha: float = 0.8,
        frame_stack: int = 1,
        # Training settings
        total_episodes: int = 1000,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        checkpoint_interval: int = 100,
        seed: int = 0,
        # Scoring
        cost_threshold: float = 25.0,
        lambda_cost: float = 1.0,
    ):
        self.controller = controller
        self.controller_name = controller_name
        self.log_dir = os.path.join(log_dir, controller_name)
        self.normalize_obs = normalize_obs
        self.smooth_actions = smooth_actions
        self.action_alpha = action_alpha
        self.frame_stack = frame_stack
        self.total_episodes = total_episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed
        self.cost_threshold = cost_threshold
        self.lambda_cost = lambda_cost

        os.makedirs(self.log_dir, exist_ok=True)
        set_global_seed(seed)

        self.logger = EpisodeLogger(self.log_dir)
        self.writer = ScalarWriter(os.path.join(self.log_dir, "tb"))

        self.reward_meter = AverageMeter(window=100)
        self.cost_meter = AverageMeter(window=100)

    def _make_train_env(self):
        return make_env(
            normalize_obs=self.normalize_obs,
            smooth_actions=self.smooth_actions,
            action_alpha=self.action_alpha,
            frame_stack=self.frame_stack,
            update_obs_stats=True,  # update stats during training
        )

    def _make_eval_env(self):
        env = make_env(
            normalize_obs=self.normalize_obs,
            smooth_actions=False,           # no smoothing during evaluation
            frame_stack=self.frame_stack,
            update_obs_stats=False,         # freeze stats during evaluation
        )
        # Copy normalizer stats from training env if available
        if hasattr(self._train_env, "rms") and hasattr(env, "rms"):
            env.rms.mean = self._train_env.rms.mean.copy()
            env.rms.var = self._train_env.rms.var.copy()
            env.rms.count = self._train_env.rms.count
        return env

    def run(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.controller_name}")
        print(f"  Episodes: {self.total_episodes}")
        print(f"  Log dir : {self.log_dir}")
        print(f"{'='*60}\n")

        self._train_env = self._make_train_env()
        t_start = time.time()

        for episode in range(self.total_episodes):
            ep_seed = self.seed + episode  # deterministic but varied seeds

            # ----- Training episode -----
            obs, _ = self._train_env.reset(seed=ep_seed)
            self.controller.reset(seed=ep_seed)

            total_reward = 0.0
            total_cost = 0.0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action, ctrl_info = self.controller.act(obs)
                obs, reward, cost, terminated, truncated, info = self._train_env.step(action)

                total_reward += reward
                total_cost += cost
                steps += 1

                # -----------------------------------------------------------
                # If your controller is a learning-based method, insert
                # the update call here, e.g.:
                #
                #   self.controller.store_transition(obs, action, reward, cost,
                #                                   next_obs, terminated)
                #   if len(self.controller.buffer) >= batch_size:
                #       loss_info = self.controller.update()
                # -----------------------------------------------------------

            # ----- Logging -----
            self.reward_meter.update(total_reward)
            self.cost_meter.update(total_cost)
            self.logger.log(total_reward, total_cost, steps)

            self.writer.add("train/reward", total_reward, episode)
            self.writer.add("train/cost", total_cost, episode)
            self.writer.add("train/reward_avg100", self.reward_meter.avg, episode)
            self.writer.add("train/cost_avg100", self.cost_meter.avg, episode)
            self.writer.add("train/episode_length", steps, episode)

            if (episode + 1) % 10 == 0:
                print_stats(
                    episode + 1,
                    self.reward_meter.avg,
                    self.cost_meter.avg,
                    steps,
                    extra={"R_last": total_reward, "C_last": total_cost},
                )

            # ----- Periodic evaluation -----
            if (episode + 1) % self.eval_interval == 0:
                print(f"\n--- Evaluation @ episode {episode+1} ---")
                eval_metrics = evaluate_policy(
                    controller=self.controller,
                    n_episodes=self.eval_episodes,
                    seeds=list(range(5000, 5000 + self.eval_episodes)),
                    normalize_obs=self.normalize_obs,
                    frame_stack=self.frame_stack,
                    cost_threshold=self.cost_threshold,
                    lambda_cost=self.lambda_cost,
                    verbose=True,
                )
                self.writer.add("eval/mean_reward", eval_metrics.mean_reward, episode)
                self.writer.add("eval/mean_cost", eval_metrics.mean_cost, episode)
                self.writer.add("eval/combined_score", eval_metrics.combined_score, episode)
                self.writer.add("eval/safe_rate", eval_metrics.safe_episode_rate, episode)
                print_report(eval_metrics, controller_name=self.controller_name)

            # ----- Checkpointing -----
            if (episode + 1) % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    self.log_dir, f"checkpoint_ep{episode+1}.pkl"
                )
                if hasattr(self.controller, "save"):
                    self.controller.save(ckpt_path)
                    print(f"[Checkpoint] Saved → {ckpt_path}")

                # Save obs normalizer stats so evaluation can reload them
                if hasattr(self._train_env, "save_stats"):
                    stats_path = os.path.join(self.log_dir, "obs_stats.npz")
                    self._train_env.save_stats(stats_path)

        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed/60:.1f} min.")

        self._train_env.close()
        self.logger.close()
        self.writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a controller for Project 1")
    p.add_argument("--controller", type=str, default="random",
                   help="Controller name: 'random' or your custom class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--action-alpha", type=float, default=0.8)
    p.add_argument("--cost-threshold", type=float, default=25.0)
    p.add_argument("--lambda-cost", type=float, default=1.0)
    p.add_argument("--no-obs-norm", action="store_true")
    p.add_argument("--no-action-smooth", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------
    # Controller selection
    # Import your own controllers here and select based on args.controller
    # ------------------------------------------------------------------

    # Minimal placeholder controller (random policy)
    # Replace this block with your actual controller import.
    class _RandomController:
        """Placeholder: random policy for smoke testing the scaffold."""
        def __init__(self, action_space):
            self.action_space = action_space

        def reset(self, seed=None):
            pass

        def act(self, observation):
            return self.action_space.sample(), {}

    # Build a temporary env to get the action space
    _tmp_env = make_env(normalize_obs=False, smooth_actions=False)
    _action_space = _tmp_env.action_space
    _tmp_env.close()

    if args.controller == "random":
        controller = _RandomController(_action_space)
    else:
        raise NotImplementedError(
            f"Unknown controller '{args.controller}'. "
            "Add your controller import here."
        )

    trainer = Trainer(
        controller=controller,
        controller_name=args.controller,
        log_dir=args.log_dir,
        normalize_obs=not args.no_obs_norm,
        smooth_actions=not args.no_action_smooth,
        action_alpha=args.action_alpha,
        frame_stack=args.frame_stack,
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        cost_threshold=args.cost_threshold,
        lambda_cost=args.lambda_cost,
    )

    trainer.run()
