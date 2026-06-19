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
        self.last_steering = 0.0
        self.last_steering = 0.0  # For smooth steering transitions during reverse

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

    def reset(self, seed=None):
        self.last_steering = 0.0

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

        # normalize goal angle to [-pi, pi]
        goal_angle_normalized = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        angle_abs = np.abs(goal_angle_normalized)

        # Compare front vs rear sector lidar to decide whether to hit with rear
        angles = bin_angles
        # widen front/rear sectors so rear is detected more easily (front/rear ±60°)
        front_mask = np.abs(angles) <= (np.pi / 3)
        rear_mask = np.abs(np.abs(angles) - np.pi) <= (np.pi / 3)
        front_max = float(np.max(sensors["goal_lidar"][front_mask]))
        rear_max = float(np.max(sensors["goal_lidar"][rear_mask]))

        hit_with_rear = False
        # relax thresholds so more targets count as 'rear-closer'
        REAR_STRONGER_FACTOR = 0.95
        REAR_MIN_DIST = 0.05
        REVERSE_ANGLE_THRESH = np.pi / 2

        use_reverse = (rear_max > front_max * REAR_STRONGER_FACTOR and rear_max > REAR_MIN_DIST) or angle_abs > REVERSE_ANGLE_THRESH

        if use_reverse:
            # Prefer reversing when the reverse direction is closer to the target
            if rear_max > front_max * REAR_STRONGER_FACTOR and rear_max > REAR_MIN_DIST:
                hit_with_rear = True
            velocity = -4.0
            # rear-facing angle (flip by pi)
            if goal_angle_normalized > 0:
                rear_angle = goal_angle_normalized - np.pi
            else:
                rear_angle = goal_angle_normalized + np.pi
            steer_target = np.clip(rear_angle, -0.785, 0.785)
            steering = 0.75 * steer_target + 0.25 * self.last_steering
            steering = np.clip(steering, -0.785, 0.785)
        else:
            # Goal is in front, move forward
            steering = np.clip(goal_angle, -0.785, 0.785)

            if np.abs(goal_angle) > 0.5:
                velocity = 2.0  # Slow down to turn effectively
            elif closest_goal > 0.6:
                velocity = 3.0  # Slow down when near the goal to prevent overshooting
            else:
                velocity = 7.0  # Speed up when heading is aligned and goal is far

        self.last_steering = steering
        action = np.array([velocity, steering], dtype=np.float32)
        
        info = {
            "policy": "scripted_move_to_goal",
            "goal_bin": int(best_bin),
            "goal_angle": float(goal_angle),
            "closest_goal_lidar": float(np.max(sensors["goal_lidar"])),
            "closest_hazard_lidar": float(np.max(sensors["hazards_lidar"])),
        }
        return action, info


class SACController(Controller):
    """
    Inference-time wrapper for a trained Stable-Baselines3 SAC policy.

    This controller does NOT train SAC. It only loads a previously trained
    model from disk and uses it to produce actions during evaluation.
    """

    def __init__(self, model_path: str):
        from stable_baselines3 import SAC
        self.model_path = model_path
        self.model = SAC.load(model_path)

    def reset(self, seed=None):
        pass

    def act(self, observation: np.ndarray):
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        info = {
            "policy": "sac",
            "model_path": self.model_path,
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
