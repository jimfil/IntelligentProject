import numpy as np


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



class ScriptedController(Controller):
    """
    Scripted Controller that parses the observation vector into individual sensors.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def _parse_obs(self, observation: np.ndarray):
        # If frame stacking is enabled, the most recent frame (76 dimensions) 
        # is always located at the end of the flattened observation vector.
        latest_obs = observation[-76:]
        
        return {
            "accelerometer": latest_obs[0:3],
            "velocimeter": latest_obs[3:6],
            "gyro": latest_obs[6:9],
            "magnetometer": latest_obs[9:12],
            "buttons_lidar": latest_obs[12:28],
            "goal_lidar": latest_obs[28:44],
            "hazards_lidar": latest_obs[44:60],
            "gremlins_lidar": latest_obs[60:76],
        }

    def act(self, observation: np.ndarray):
        sensors = self._parse_obs(observation)
        
        # 1. Find the bin with the highest reading (where the goal is closest/strongest)
        best_bin = np.argmax(sensors["goal_lidar"])
        
        # 2. Map bin index (0 to 15) to a relative angle in radians [-pi, pi]
        # Bin 0 is 0 (directly in front), bin 4 is pi/2 (left), bin 12 is -pi/2 (right)
        bin_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        bin_angles = (bin_angles + np.pi) % (2 * np.pi) - np.pi
        goal_angle = bin_angles[best_bin]
        
        # 3. Control velocity & steering: move forward, or reverse if the goal is behind
        closest_goal = np.max(sensors["goal_lidar"])
        
        if np.abs(goal_angle) > 2.0:
            # Goal is behind, so we reverse
            velocity = -4.0
            # Align the rear of the car with the goal
            if goal_angle > 0:
                rear_angle = goal_angle - np.pi
            else:
                rear_angle = goal_angle + np.pi
            steering = np.clip(rear_angle, -0.785, 0.785)
        else:
            # Goal is in front, move forward
            steering = np.clip(goal_angle, -0.785, 0.785)
            
            if np.abs(goal_angle) > 0.5:
                velocity = 2.0  # Slow down to turn effectively
            elif closest_goal > 0.6:
                velocity = 3.0  # Slow down when near the goal to prevent overshooting
            else:
                velocity = 7.0  # Speed up when heading is aligned and goal is far
            
        action = np.array([velocity, steering], dtype=np.float32)
        
        info = {
            "policy": "scripted_move_to_goal",
            "goal_bin": int(best_bin),
            "goal_angle": float(goal_angle),
            "closest_goal_lidar": float(np.max(sensors["goal_lidar"])),
            "closest_hazard_lidar": float(np.max(sensors["hazards_lidar"])),
        }
        return action, info



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
