import sys
import os
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import environment_setup
from environment_setup import make_env, ObsNormWrapper
from controllers import ToController
import safety_gymnasium

# Hardcoded artifact directory for saving the videos
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\cbe383ed-e023-4ae7-b215-1c28ca8eb17d"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def make_env_rgb_high_res(
    normalize_obs: bool = True,
    obs_stats_path: str = None,
    camera_name: str = "fixedfar"
):
    env = safety_gymnasium.make("SafetyRacecarButton2-v0", render_mode="rgb_array")
    
    from environment_setup import CostTrackingWrapper, ObsNormWrapper
    env = CostTrackingWrapper(env)
    
    if normalize_obs:
        env = ObsNormWrapper(env, update=False)
        if obs_stats_path and os.path.exists(obs_stats_path):
            env.load_stats(obs_stats_path)
            print(f"[Record] Loaded observation stats from {obs_stats_path}")
            
    # Set the high resolution parameters
    env.unwrapped.render_parameters.width = 1024
    env.unwrapped.render_parameters.height = 1024
    env.unwrapped.render_parameters.camera_name = camera_name
    print(f"[Record] Initialized {camera_name} view at 1024x1024 resolution.")
    
    return env

def overlay_info(frame, step, reward, cost, seed, status_text, camera_name):
    # Convert numpy array to PIL Image (frame is already 1024x1024)
    img = Image.fromarray(frame)
    
    draw = ImageDraw.Draw(img)
    # Draw a semi-transparent black rectangle at the top for HUD
    draw.rectangle([0, 0, 1024, 110], fill=(0, 0, 0))
    
    try:
        # Load standard Arial font at size 28 for crisp readability
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
        
    text_line1 = f"Seed: {seed} | Step: {step} | POV: {camera_name.upper()} View | Controller: Simple PPO"
    text_line2 = f"Cumulative Reward: {reward:.3f} | Cost: {cost:.1f} ({status_text})"
    
    # Use appropriate color encoding
    color_line1 = (255, 255, 255)
    if "SUCCESS" in status_text:
        color_line2 = (100, 255, 100)
    elif "FAILURE" in status_text:
        color_line2 = (255, 100, 100)
    else:
        color_line2 = (255, 255, 255)
        
    draw.text((20, 15), text_line1, fill=color_line1, font=font)
    draw.text((20, 60), text_line2, fill=color_line2, font=font)
    
    return np.array(img)

def run_and_record_episode(controller, env, seed, video_paths, status_text, camera_name):
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)
    
    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    frames = []
    
    # Initial frame
    frame = env.render()
    if frame is not None:
        frames.append(overlay_info(frame, steps, total_reward, total_cost, seed, status_text, camera_name))
        
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = controller.act(obs)
        obs, reward, cost, terminated, truncated, step_info = env.step(action)
        
        total_reward += reward
        total_cost += cost
        steps += 1
        
        frame = env.render()
        if frame is not None:
            frames.append(overlay_info(frame, steps, total_reward, total_cost, seed, status_text, camera_name))
            
    # Save video to all requested paths
    for path in video_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving video to {path}...")
        writer = imageio.get_writer(path, fps=30)
        for f in tqdm(frames, desc=f"Writing {os.path.basename(path)}"):
            writer.append_data(f)
        writer.close()
        
    return total_reward, total_cost, steps

def main():
    model_path = os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/best_model.zip")
    obs_stats_path = os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/obs_stats.npz")
    
    print(f"Loading controller from {model_path}...")
    controller = ToController(model_path)
    
    success_seed = 0
    failure_seed = 9
    
    # ----------------- 1. Record Birds-Eye POV (fixedfar) -----------------
    env_birds_eye = make_env_rgb_high_res(
        normalize_obs=True, 
        obs_stats_path=obs_stats_path, 
        camera_name="fixedfar"
    )
    
    success_paths_be = [
        os.path.join(ARTIFACT_DIR, "ppo_success_birds_eye.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/success_demo_birds_eye.mp4")
    ]
    failure_paths_be = [
        os.path.join(ARTIFACT_DIR, "ppo_failure_birds_eye.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/failure_demo_birds_eye.mp4")
    ]
    
    print("\nRecording Success Birds-Eye Episode...")
    run_and_record_episode(controller, env_birds_eye, success_seed, success_paths_be, "SAFE - SUCCESS", "fixedfar")
    
    print("\nRecording Failure Birds-Eye Episode...")
    run_and_record_episode(controller, env_birds_eye, failure_seed, failure_paths_be, "UNSAFE - FAILURE", "fixedfar")
    
    env_birds_eye.close()
    
    # ----------------- 2. Record First-Person POV (vision) -----------------
    env_first_person = make_env_rgb_high_res(
        normalize_obs=True, 
        obs_stats_path=obs_stats_path, 
        camera_name="vision"
    )
    
    success_paths_fp = [
        os.path.join(ARTIFACT_DIR, "ppo_success.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/success_demo.mp4")
    ]
    failure_paths_fp = [
        os.path.join(ARTIFACT_DIR, "ppo_failure.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/failure_demo.mp4")
    ]
    
    print("\nRecording Success First-Person Episode...")
    run_and_record_episode(controller, env_first_person, success_seed, success_paths_fp, "SAFE - SUCCESS", "vision")
    
    print("\nRecording Failure First-Person Episode...")
    run_and_record_episode(controller, env_first_person, failure_seed, failure_paths_fp, "UNSAFE - FAILURE", "vision")
    
    env_first_person.close()
    
    print("\nFinished recording all high-resolution episodes successfully!")

if __name__ == "__main__":
    main()
