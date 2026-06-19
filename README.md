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

## Installation

To ensure compatibility and resolve package version conflicts (especially on Python 3.11+), install the dependencies in this specific order:

```bash
pip install pygame mujoco==2.3.3 gymnasium-robotics==1.2.2 xmltodict pyyaml
pip install safety-gymnasium --no-deps
pip install gymnasium==0.28.1 numpy==1.23.5 torch>=2.0.0 matplotlib>=3.7.0 tensorboard>=2.13.0 imageio>=2.31.0 imageio-ffmpeg>=0.4.8 tqdm>=4.65.0 pandas>=2.0.0
```

## Quick Start

```bash
# Test the pipeline with a random controller
python train.py --controller random --episodes 5

# Evaluate
python evaluation.py
```

## Implementing Your Controller

Edit `controllers.py` → `TeamController`:

```python
class TeamController(Controller):
    def reset(self, seed=None):
        # reset internal state (hidden states, buffers, etc.)
        pass

    def act(self, observation):
        # return (action, info)
        # action shape: (2,) — [rear_wheel_vel, front_steering]
        ...
```

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
