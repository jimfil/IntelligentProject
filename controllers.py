"""
Project 1: Safe Mobile Robot Button Navigation
Controller interface and placeholder classes.

Each team must implement its own controller here.
This file defines the base interface and provides two minimal examples:

  1. RandomController   — random policy (simple baseline)
  2. BaseController     — stub that teams must fill in

The benchmark interface requires:

    class Controller:
        def reset(self, seed=None):
            ...
        def act(self, observation):
            return action, info   # action: np.ndarray, info: dict

The 'info' dict may be used for logging/visualization only.
It MUST NOT modify the environment, reward, cost, or termination.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class Controller:
    """
    Benchmark controller interface.

    All submitted controllers must inherit from this class and implement
    reset() and act().
    """

    def reset(self, seed=None):
        """
        Called at the start of each episode.
        Use this to reinitialize any internal state (e.g. recurrent hidden
        states, memory buffers, internal clocks).

        Parameters
        ----------
        seed : int or None
            Episode seed, provided for reproducibility.
        """
        pass

    def act(self, observation: np.ndarray):
        """
        Selects an action given the current observation.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation (after any allowed wrappers).

        Returns
        -------
        action : np.ndarray
            Action to execute in the environment.
        info : dict
            Diagnostic information for logging/visualization only.
            Must NOT affect the environment or scoring.
        """
        raise NotImplementedError("Subclasses must implement act().")


# ---------------------------------------------------------------------------
# Simple baseline 1: Random Policy
# ---------------------------------------------------------------------------

class RandomController(Controller):
    """
    Random policy — samples uniformly from the action space.

    This serves as the simplest baseline to verify the pipeline works.
    It should perform poorly on both reward and safety metrics.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: np.ndarray):
        action = self.action_space.sample()
        return action, {"policy": "random"}


# ---------------------------------------------------------------------------
# Team controller stub — FILL THIS IN
# ---------------------------------------------------------------------------

class TeamController(Controller):
    """
    Team's main controller — implement your approach here.

    You may use:
      - Classical control  (PID, LQR, MPC, potential fields, etc.)
      - Safe RL            (CPO, PPO-Lagrangian, TRPO-Lagrangian, etc.)
      - Standard RL        (PPO, SAC, TD3, etc.)
      - Hybrid approaches

    Allowed techniques (benchmark rules):
      - Observation normalization (handled by environment_setup.py wrappers)
      - Action smoothing          (handled by environment_setup.py wrappers)
      - Frame stacking            (handled by environment_setup.py wrappers)
      - Recurrent state (e.g. LSTM hidden state — reset in reset())
      - Memory buffers (replay buffer, demonstrations)
      - Learned representations
      - Model-based planning

    Forbidden (benchmark rules):
      - Modifying reward, cost, termination logic, random seeds, or
        the scoring script.
    """

    def __init__(self):
        # Initialize your model, policy network, parameters, etc.
        # Example:
        #   self.policy = MyPolicyNetwork(obs_dim=..., act_dim=2)
        #   self.policy.load("checkpoints/best_policy.pt")
        pass

    def reset(self, seed=None):
        # Reset internal state at the start of each episode.
        # Example for a recurrent policy:
        #   self.hidden_state = torch.zeros(1, self.hidden_size)
        pass

    def act(self, observation: np.ndarray):
        # TODO: implement your controller logic here.
        # Return (action, info) where action is a numpy array compatible
        # with the environment's action space:
        #   action[0] — rear-wheel velocity  ∈ [-1, 1]
        #   action[1] — front-wheel steering ∈ [-1, 1]
        raise NotImplementedError(
            "TeamController.act() is not implemented. "
            "Replace this stub with your actual controller."
        )
