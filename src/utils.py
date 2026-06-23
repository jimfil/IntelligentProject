"""
Project 1: Safe Mobile Robot Button Navigation
Logging, video recording, and general utilities.
"""

import os
import time
import json
import csv
from collections import defaultdict
from typing import Optional, List, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Episode statistics logger
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """
    Accumulates per-episode metrics and writes them to a CSV file.

    Tracked metrics per episode:
        - episode index
        - total reward
        - total cost
        - episode length (steps)
        - wall-clock time (seconds)
    """

    FIELDS = ["episode", "reward", "cost", "length", "time_s"]

    def __init__(self, log_dir: str, filename: str = "episodes.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._episode_idx = 0
        self._t0 = time.time()

    def log(self, reward: float, cost: float, length: int):
        row = {
            "episode": self._episode_idx,
            "reward": round(reward, 6),
            "cost": round(cost, 6),
            "length": length,
            "time_s": round(time.time() - self._t0, 2),
        }
        self._writer.writerow(row)
        self._file.flush()
        self._episode_idx += 1

    def close(self):
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Running average meter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Keeps a sliding-window average of a scalar metric."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: List[float] = []

    def update(self, value: float):
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)

    @property
    def avg(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def last(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __repr__(self):
        return f"AverageMeter(avg={self.avg:.4f}, n={len(self._values)})"


# ---------------------------------------------------------------------------
# TensorBoard-style scalar writer (plain JSON fallback if TB not available)
# ---------------------------------------------------------------------------

class ScalarWriter:
    """
    Writes scalar metrics for visualization.
    Uses TensorBoard SummaryWriter when available, otherwise falls back to JSON.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._scalars: Dict[str, List] = defaultdict(list)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=log_dir)
        except ImportError:
            self._tb = None

    def add(self, tag: str, value: float, step: int):
        self._scalars[tag].append((step, value))
        if self._tb is not None:
            self._tb.add_scalar(tag, value, global_step=step)

    def flush(self):
        # Save JSON snapshot for non-TB viewers
        json_path = os.path.join(self.log_dir, "scalars.json")
        with open(json_path, "w") as f:
            json.dump(self._scalars, f, indent=2)
        if self._tb is not None:
            self._tb.flush()

    def close(self):
        self.flush()
        if self._tb is not None:
            self._tb.close()


# ---------------------------------------------------------------------------
# Video recorder
# ---------------------------------------------------------------------------

class VideoRecorder:
    """
    Records environment frames to an MP4 file using imageio.

    Usage:
        rec = VideoRecorder("videos/run_001.mp4", fps=30)
        rec.add_frame(frame_rgb_array)
        ...
        rec.close()
    """

    def __init__(self, path: str, fps: int = 30):
        import imageio
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.writer = imageio.get_writer(path, fps=fps)
        self.path = path
        self.frame_count = 0

    def add_frame(self, frame: np.ndarray):
        """Append an RGB frame (H x W x 3, uint8)."""
        self.writer.append_data(frame)
        self.frame_count += 1

    def close(self):
        self.writer.close()
        print(f"[VideoRecorder] Saved {self.frame_count} frames → {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    """Sets random seeds for reproducibility across numpy, random, and torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Pretty console printer
# ---------------------------------------------------------------------------

def print_stats(episode: int, reward: float, cost: float, length: int,
                extra: Optional[Dict] = None):
    """Prints a formatted line of episode statistics."""
    line = (
        f"[Ep {episode:>5d}]  "
        f"R={reward:+8.3f}  "
        f"C={cost:7.3f}  "
        f"L={length:>4d}"
    )
    if extra:
        for k, v in extra.items():
            line += f"  {k}={v:.4f}" if isinstance(v, float) else f"  {k}={v}"
    print(line)


if __name__ == "__main__":
    meter = AverageMeter(window=5)
    for v in [1, 2, 3, 4, 5, 6]:
        meter.update(v)
    print("AverageMeter test:", meter)
