# Running the Project in Docker

This project is containerized using Docker and Docker Compose, allowing you to train, evaluate, and monitor agents in a consistent environment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

## 1. Build the Containers

Build the Docker images first:

```bash
docker compose build
```

---

## 2. Monitor with TensorBoard

Start the TensorBoard service. It runs in the background and exposes port `6006`:

```bash
docker compose up tensorboard
```

Open your browser and navigate to: [http://localhost:6006](http://localhost:6006)

---

## 3. Train Agents

### Soft Actor-Critic (SAC) Baseline
To start training the baseline SAC policy:

```bash
docker compose run --rm train-sac
```

### PPO-Lagrangian Safety Policy
To start training the PPO-Lagrangian policy:

```bash
docker compose run --rm train-ppo-lagrangian
```

*Note: The model checkpoints, tensorboard logs, and normalization stats will be written back to the `runs/` folder on your host machine.*

---

## 4. Evaluate Policies

You can evaluate different policies using the `evaluate` service. By default, it runs the `scripted` controller for 20 episodes.

### Default Evaluation (Scripted Controller)
```bash
docker compose run --rm evaluate
```

### Evaluate SAC Controller
To override the default controller, append arguments to the command:

```bash
docker compose run --rm evaluate python src/evaluation.py --controller sac --n-episodes 20
```

### Evaluate PPO-Lagrangian Controller
```bash
docker compose run --rm evaluate python src/evaluation.py --controller ppo_lagrangian --n-episodes 20
```

---

## 5. Interactive Debugging / Custom Commands

If you want to run custom scripts, check file systems, or run interactive bash sessions inside the container:

```bash
docker compose run --rm shell
```

This starts a shell where all code changes you make on the host are instantly shared, and any generated artifacts (plots, logs, models) are written directly to your host directory.
