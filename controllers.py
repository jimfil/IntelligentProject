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

    def __init__(self):
        self.last_steering = 0.0  # For smooth steering transitions during reverse
        self.is_reversing = False

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
        self.is_reversing = False

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
        front_mask = np.abs(angles) <= (np.pi / 3)
        rear_mask = np.abs(np.abs(angles) - np.pi) <= (np.pi / 3)
        front_max = float(np.max(sensors["goal_lidar"][front_mask]))
        rear_max = float(np.max(sensors["goal_lidar"][rear_mask]))

        # Identify the active goal bin via argmax on goal_lidar — works with normalized obs.
        # Zero exactly that bin in buttons_lidar so non-goal buttons (including the
        # previously activated one) remain visible as obstacles.
        wrong_buttons = sensors["buttons_lidar"].copy()
        goal_bin_idx = int(np.argmax(sensors["goal_lidar"]))
        wrong_buttons[goal_bin_idx] = 0.0  # suppress current-goal bin only

        obstacle_lidar = np.maximum.reduce([
            sensors["hazards_lidar"],
            sensors["gremlins_lidar"],
            wrong_buttons,
        ])

        # Define front and rear sectors for obstacle scanning
        front_indices = [13, 14, 15, 0, 1, 2, 3]
        rear_indices = [5, 6, 7, 8, 9, 10, 11]
        
        front_obstacle_val = np.max(obstacle_lidar[front_indices])
        rear_obstacle_val = np.max(obstacle_lidar[rear_indices])

        # Two thresholds: buttons are smaller than hazards, their lidar signal peaks
        # at a lower value for the same proximity — give them a separate, lower threshold.
        HAZARD_THRESH = 0.85  # for hazards & gremlins
        BUTTON_THRESH = 0.70  # for wrong (non-goal) buttons

        hazard_only = np.maximum(sensors["hazards_lidar"], sensors["gremlins_lidar"])
        front_avoid = (np.max(hazard_only[front_indices]) > HAZARD_THRESH or
                       np.max(wrong_buttons[front_indices]) > BUTTON_THRESH)
        rear_avoid  = (np.max(hazard_only[rear_indices])  > HAZARD_THRESH or
                       np.max(wrong_buttons[rear_indices])  > BUTTON_THRESH)

        # Gear selection:
        # 1. Maneuver threshold: only reverse if the target is close (closest_goal > 0.35)
        # 2. Opposite direction threshold: if the target is behind us (angle_abs > 1.8)
        MANEUVER_DIST_THRESH = 0.35
        REVERSE_ANGLE_THRESH = 1.8  # ~103 degrees
        
        use_reverse = (angle_abs > REVERSE_ANGLE_THRESH)

        if use_reverse:
            # Calculate rear-relative goal angle
            if goal_angle_normalized > 0:
                rear_angle = goal_angle_normalized - np.pi
            else:
                rear_angle = goal_angle_normalized + np.pi

            if rear_avoid:
                # Find the angle of the closest rear obstacle
                closest_rear_bin = rear_indices[np.argmax(obstacle_lidar[rear_indices])]
                obstacle_angle = bin_angles[closest_rear_bin]
                # Flip angle to be rear-relative
                rear_obs_angle = obstacle_angle - np.pi if obstacle_angle > 0 else obstacle_angle + np.pi
                
                if np.abs(rear_obs_angle) < 0.1:  # Directly behind, steer to the safer side
                    left_obs = np.max(obstacle_lidar[[5, 6, 7]])
                    right_obs = np.max(obstacle_lidar[[9, 10, 11]])
                    avoid_steer = 0.785 if left_obs < right_obs else -0.785
                else:
                    # Steer rear away from obstacle
                    avoid_steer = -0.785 if rear_obs_angle > 0 else 0.785
                
                steer_target = 0.7 * avoid_steer + 0.3 * np.clip(-rear_angle, -0.785, 0.785)
                velocity = -2.0  # Slow down in reverse to turn
            else:
                # No obstacle behind, normal reverse steering towards the target
                velocity = -5.0
                steer_target = np.clip(-rear_angle, -0.785, 0.785)
            
            # Smoothed steering for reverse to avoid spinning out
            steering = 0.8 * steer_target + 0.2 * self.last_steering
            steering = np.clip(steering, -0.785, 0.785)
        else:
            if front_avoid:
                # Find the angle of the closest front obstacle
                closest_front_bin = front_indices[np.argmax(obstacle_lidar[front_indices])]
                obstacle_angle = bin_angles[closest_front_bin]
                
                if np.abs(obstacle_angle) < 0.1:  # Directly in front, steer to the safer side
                    left_obs = np.max(obstacle_lidar[[1, 2, 3]])
                    right_obs = np.max(obstacle_lidar[[13, 14, 15]])
                    avoid_steer = 0.785 if left_obs < right_obs else -0.785
                else:
                    # Steer away from obstacle
                    avoid_steer = -0.785 if obstacle_angle > 0 else 0.785
                
                steer_target = 0.7 * avoid_steer + 0.3 * np.clip(goal_angle, -0.785, 0.785)
                velocity = 2.0  # Slow down forward velocity to turn
                steering = 0.75 * steer_target + 0.25 * self.last_steering
                steering = np.clip(steering, -0.785, 0.785)
            else:
                # No obstacle in front, original forward driving towards the target (unsmoothed, non-straightening steering)
                if goal_angle > 0.15:
                    steering = 0.785
                elif goal_angle < -0.15:
                    steering = -0.785
                else:
                    steering = goal_angle
                    
                if np.abs(goal_angle) > 0.5:
                    velocity = 2.0  # Slow down to turn effectively
                elif closest_goal > 0.6:
                    velocity = 3.0  # Slow down near goal
                else:
                    velocity = 6.0  # Cruise speed

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


class ToController(Controller):
    """
    Inference-time wrapper for a trained Stable-Baselines3 PPO policy.
    """

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO
        self.model_path = model_path
        self.model = PPO.load(model_path)

    def reset(self, seed=None):
        pass

    def act(self, observation: np.ndarray):
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        info = {
            "policy": "ppo_lagrangian",
            "model_path": self.model_path,
        }
        return action, info

