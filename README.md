# Project 1: Safe Mobile Robot Button Navigation

**Course:** Intelligent Control 2025-2026  
**Environment:** `SafetyRacecarButton2-v0` (Safety-Gymnasium)  
**Robot:** Racecar — rear-wheel velocity + front-wheel steering control

---

## Project Structure

```
project1_safe_nav/
├── environment_setup.py   # env factory, wrappers (obs norm, action smooth, frame stack)
├── evaluation.py          # evaluation loop, metrics, comparison table
├── train.py               # training scaffold (controller-agnostic)
├── controllers.py         # Controller interface + team stub (FILL IN)
├── utils.py               # logging, video recording, seeds
├── requirements.txt       # dependencies
└── README.md
```

## Benchmark Rules (Reminder)

**FORBIDDEN** — Do NOT modify:
- Reward function
- Cost function
- Termination logic
- Random seeds used for official evaluation
- Scoring script

**ALLOWED** — You may use:
- Observation normalization ✓ (`ObsNormWrapper` in `environment_setup.py`)
- Action smoothing ✓ (`ActionSmoothingWrapper` in `environment_setup.py`)
- Frame stacking ✓ (`FrameStackWrapper` in `environment_setup.py`)
- Recurrent state, memory buffers, learned representations ✓
- Model-based planning, demonstrations ✓

## Setup and Installation

Create the virtual environment and activate it. This isolates your project packages and ensures standard paths are used.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# On Linux or macOS:
source venv/bin/activate
```

Install the dependencies in this specific order. This exact ordering is required to resolve version conflicts between Safety-Gymnasium, Gymnasium, and MuJoCo on Python 3.11+.

```bash
# 1. Install MuJoCo physics simulation environment and graphics packages
pip install pygame mujoco==2.3.3 gymnasium-robotics==1.2.2 xmltodict pyyaml

# 2. Install Safety-Gymnasium without pulling standard gymnasium dependencies to avoid version issues
pip install safety-gymnasium --no-deps

# 3. Install core libraries for reinforcement learning, neural networks, and evaluation
pip install gymnasium==0.28.1 numpy==1.23.5 torch>=2.0.0 matplotlib>=3.7.0 tensorboard>=2.13.0 imageio>=2.31.0 imageio-ffmpeg>=0.4.8 tqdm>=4.65.0 pandas>=2.0.0 stable-baselines3==2.1.0
```

---

## Training Policies

Train the baseline SAC (Soft Actor-Critic) agent. This model optimizes solely for task rewards, ignoring safety costs. It will write checkpoint logs and observation normalization statistics under `runs/sac_baseline/`.

```bash
python train_sac.py --normalize-obs --total-timesteps 300000
```

Train the safety-aware PPO-Lagrangian agent. This model dynamically balances task rewards and safety constraints to keep safety costs under the threshold of 25.0. It will save final checkpoints and statistics under `runs/ppo_lagrangian/`.

```bash
python train_ppo_lagrangian.py --total-timesteps 300000
```

---

## Evaluating Controllers

Evaluate the Scripted Controller baseline. It does not use observation normalization and executes heuristic-based navigation.

```bash
python evaluation.py --controller scripted --n-episodes 20
```

Evaluate the trained SAC Baseline agent. This will load the best model and corresponding normalization statistics (`obs_stats.npz`) from the `runs/sac_baseline/` folder.

```bash
python evaluation.py --controller sac --n-episodes 20
```

Evaluate the trained safety-aware PPO-Lagrangian agent. This loads the policy weights and corresponding normalization statistics from `runs/ppo_lagrangian/`.

```bash
python evaluation.py --controller ppo_lagrangian --n-episodes 20
```

Run evaluation with rendering enabled to visualize the racecar's driving behavior, avoidance maneuvers, and interactions with buttons:

```bash
python evaluation.py --controller ppo_lagrangian --n-episodes 5 --render
```

---

## Monitoring Progress

Launch TensorBoard to track rewards, safety costs, and Lagrangian constraint multiplier value curves in real time:

```bash
tensorboard --logdir runs
```

---

## Observation & Action Space

```
SafetyRacecarButton2-v0
  obs : continuous (varies with wrappers/frame stacking)
  act : Box(2,)  — [rear-wheel velocity, front-wheel steering angle]
  step returns: obs, reward, COST, terminated, truncated, info
```

## Scoring Metrics

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average cumulative reward per episode |
| `mean_cost` | Average cumulative safety cost per episode |
| `safe_episode_rate` | Fraction of episodes with cost ≤ threshold |
| `combined_score` | `mean_reward − λ × mean_cost` (primary ranking metric) |

## Required Comparison

Your final submission must compare **at least two** approaches:
1. One simple baseline (random policy, PID, scripted controller, or vanilla RL)
2. One meaningful method (PPO-Lagrangian, CPO, SAC, MPPI, or safety-aware approach)
