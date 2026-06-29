# Project 1: Safe Mobile Robot Button Navigation

**Course:** Intelligent Control 2025-2026  
**Environment:** `SafetyRacecarButton2-v0` (Safety-Gymnasium)  
**Robot:** Racecar - rear-wheel velocity + front-wheel steering control

---

## Project Structure

```
project1_safe_nav/
├── documentation/
│   ├── report.tex             # LaTeX source code of the project report and presentation
│   └── *.jpg / *.png          # visual assets for the report
├── src/
│   ├── environment_setup.py   # env factory and custom wrappers
│   ├── evaluation.py          # evaluation loop, metrics, and comparisons
│   ├── controllers.py         # controller interface and scripted controller
│   ├── train_ppo.py           # baseline PPO training script
│   ├── train_ppo_lagrangian.py# Safe PPO-Lagrangian training script
│   ├── train_ppo_lagrangian_multicpu.py # Multi-CPU PPO-Lagrangian training
│   └── record.py              # script to record high-resolution videos
├── docker-compose.yml         # container services definition
├── Dockerfile                 # base image description
├── requirements.txt           # Python package dependencies
└── README.md                  # main project guide
```

---

## Installation & Setup Options

You can run the project either locally using a Virtual Environment (venv) or containerized using Docker.

### Option 1: Local Virtual Environment (Recommended for GUI Rendering)
Use this option if you want to visualize the robot's movement in real-time (rendering the 3D physics scene).
Recommended ~ Python 3.11

#### 1. Setup the Environment
Create the virtual environment, activate it, and install the dependencies:
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

# Install the dependencies in this exact order:
# 1. Install MuJoCo physics simulation environment and graphics packages
pip install pygame mujoco==2.3.3 gymnasium-robotics==1.2.2 xmltodict pyyaml

# 2. Install Safety-Gymnasium without standard gymnasium dependencies
pip install safety-gymnasium --no-deps

# 3. Install core libraries for reinforcement learning and evaluation
pip install -r requirements.txt
```

#### 2. Train Agents
*   **Unconstrained PPO Baseline**:
    ```bash
    python src/train_ppo.py
    ```
*   **Safe PPO (PPO-Lagrangian)**:
    ```bash
    python src/train_ppo_lagrangian.py
    ```

#### 3. Evaluate Policies
*   **Scripted Controller**:
    ```bash
    python src/evaluation.py --controller scripted --n-episodes 20
    ```
*   **Evaluate Unconstrained PPO**:
    ```bash
    python src/evaluation.py --controller ppo --model-path runs/ppo_unconstrained/best_model.zip --obs-stats-path runs/ppo_unconstrained/obs_stats.npz --n-episodes 20
    ```
*   **Evaluate Safe PPO**:
    ```bash
    python src/evaluation.py --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 20
    ```
*   **Evaluate with 3D Rendering (GUI Visualizer)**:
    Add the `--render` flag to any evaluation command:
    ```bash
    python src/evaluation.py --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 5 --render
    ```

#### 4. Monitor Progress with TensorBoard
```bash
tensorboard --logdir runs
```

---

### Option 2: Docker & Docker Compose (No GUI Rendering Support)
Use this option for a quick, zero-install, reproducible headless environment. Note: **Rendering/visualizing is NOT supported** in this setup.

#### 1. Build the Containers
```bash
docker-compose build
```

#### 2. Train Agents
*   **Unconstrained PPO Baseline**:
    ```bash
    docker-compose up train-simple-ppo
    ```
*   **Safe PPO (PPO-Lagrangian)**:
    ```bash
    docker-compose up train-safe-ppo
    ```

#### 3. Evaluate Policies
*   **Scripted Controller**:
    ```bash
    docker-compose run --rm evaluation --controller scripted
    ```
*   **Evaluate Unconstrained PPO**:
    ```bash
    docker-compose run --rm evaluation --controller ppo --model-path runs/ppo_unconstrained/best_model.zip --obs-stats-path runs/ppo_unconstrained/obs_stats.npz --n-episodes 20
    ```
*   **Evaluate Safe PPO**:
    ```bash
    docker-compose run --rm evaluation --controller ppo --model-path runs/ppo_model/best_model.zip --obs-stats-path runs/ppo_model/obs_stats.npz --n-episodes 20
    ```

#### 4. Monitor Progress with TensorBoard
Exposes TensorBoard at http://localhost:6006:
```bash
docker-compose up tensorboard
```

#### 5. Interactive Debugging
To open a bash shell inside the container:
```bash
docker-compose run --rm shell
```

---

## Observation & Action Space

```
SafetyRacecarButton2-v0
  obs : continuous (varies with wrappers/frame stacking)
  act : Box(2,)  - [rear-wheel velocity, front-wheel steering angle]
  step returns: obs, reward, COST, terminated, truncated, info
```

## Scoring Metrics

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average cumulative reward per episode |
| `mean_cost` | Average cumulative safety cost per episode |
| `safe_episode_rate` | Fraction of episodes with cost ≤ threshold |
| `combined_score` | `mean_reward − λ × mean_cost` (primary ranking metric) |
