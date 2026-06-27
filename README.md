# Project 1: Safe Mobile Robot Button Navigation

**Course:** Intelligent Control 2025-2026  
**Environment:** `SafetyRacecarButton2-v0` (Safety-Gymnasium)  
**Robot:** Racecar — rear-wheel velocity + front-wheel steering control

---

## Project Structure

project1_safe_nav/
├── src/
│   ├── environment_setup.py   # env factory, wrappers (obs norm, action smooth, frame stack)
│   ├── evaluation.py          # evaluation loop, metrics, comparison table
│   └── controllers.py         # Controller interface + team stub (FILL IN)
├── requirements.txt       # dependencies
├── README.md
└── README_DOCKER.md

## Setup and Installation (using Docker)

This project is containerized using Docker and Docker Compose, allowing you to train, evaluate, and monitor agents in a consistent environment without manual installation issues.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Build the Containers
```bash
docker-compose build
```

### 2. Monitor with TensorBoard
Start the TensorBoard service. It runs in the background and exposes port `6006`:
```bash
docker-compose up tensorboard
```
Open your browser and navigate to: http://localhost:6006

### 3. Train Agents
*   **Unconstrained PPO Baseline**:
    ```bash
    docker compose run --rm train-ppo-unconstrained
    ```
*   **Safe PPO Safety Policy**:
    ```bash
    docker compose run --rm train-safe-ppo
    ```
    *Note: Checkpoints, logs, and stats are saved to the `runs/` directory.*

### 4. Evaluate Policies
*   **Default Evaluation (Scripted Controller)**:
    ```bash
    docker compose run --rm evaluation --controller scripted
    ```
*   **Evaluate Unconstrained PPO**:
    ```bash
    docker compose run --rm evaluation --controller ppo --model-path runs/ppo_unconstrained/best_model.zip --obs-stats-path runs/ppo_unconstrained/obs_stats.npz --n-episodes 20
    ```
*   **Evaluate Safe PPO**:
    ```bash
    docker compose run --rm evaluation --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 20
    ```

### 5. Interactive Debugging / Custom Commands
To start an interactive shell inside the container:
```bash
docker compose run --rm shell
```

---

## Setup and Installation (if you dont use Docker)

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
pip install -r requirements.txt
```

---

## Training Policies

Train the baseline unconstrained PPO agent. This model optimizes solely for task rewards, ignoring safety costs. It will write checkpoint logs and observation normalization statistics under `runs/ppo_unconstrained/`.

```bash
python src/train_ppo.py --normalize-obs --total-timesteps 3000000
```

Train the safety-aware PPO-Lagrangian agent. This model dynamically balances task rewards and safety constraints to keep safety costs under the threshold of 25.0. It will save final checkpoints and statistics under `runs/ppo_model/`.

```bash
python src/train_ppo_lagrangian.py --total-timesteps 3000000
```

---

## Evaluating Controllers

Evaluate the Scripted Controller baseline. It does not use observation normalization and executes heuristic-based navigation.

```bash
python src/evaluation.py --controller scripted --n-episodes 20
```

Evaluate the trained unconstrained PPO Baseline agent. This will load the best model and corresponding normalization statistics (`obs_stats.npz`) from the `runs/ppo_unconstrained/` folder.

```bash
python src/evaluation.py --controller ppo --model-path runs/ppo_unconstrained/best_model.zip --obs-stats-path runs/ppo_unconstrained/obs_stats.npz --n-episodes 20
```

Evaluate the trained safety-aware PPO-Lagrangian agent. This loads the policy weights and corresponding normalization statistics from `runs/ppo_model/`.

```bash
python src/evaluation.py --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 20
```

Run evaluation with rendering enabled to visualize the racecar's driving behavior, avoidance maneuvers, and interactions with buttons:

```bash
python src/evaluation.py --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 5 --render
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


