import sys
import os
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import environment_setup
from environment_setup import make_env, ObsNormWrapper
from controllers import ToController
import safety_gymnasium

# Hardcoded artifact directory for saving the videos
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\cbe383ed-e023-4ae7-b215-1c28ca8eb17d"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def make_env_rgb_birds_eye(
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
            
    # Set the camera view
    env.unwrapped.render_parameters.camera_name = camera_name
    print(f"[Record] Set camera view to: {camera_name}")
    
    return env

def overlay_info(frame, step, reward, cost, seed, status_text, camera_name):
    # Convert numpy array to PIL Image
    img = Image.fromarray(frame)
    # Upscale to 512x512
    img = img.resize((512, 512), Image.Resampling.NEAREST)
    
    draw = ImageDraw.Draw(img)
    # Draw a semi-transparent black rectangle at the top
    draw.rectangle([0, 0, 512, 60], fill=(0, 0, 0))
    
    # Text sizes and positions
    text_line1 = f"Seed: {seed} | Step: {step} | POV: {camera_name.upper()} View"
    text_line2 = f"Cumulative Reward: {reward:.3f} | Cost: {cost:.1f} ({status_text})"
    
    draw.text((10, 10), text_line1, fill=(255, 255, 255))
    draw.text((10, 32), text_line2, fill=(255, 255, 255) if cost <= 25.0 else (255, 100, 100))
    
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
    camera_name = "fixedfar"
    
    env_record = make_env_rgb_birds_eye(
        normalize_obs=True, 
        obs_stats_path=obs_stats_path, 
        camera_name=camera_name
    )
    
    # Paths to save
    success_paths = [
        os.path.join(ARTIFACT_DIR, "ppo_success_birds_eye.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/success_demo_birds_eye.mp4")
    ]
    
    failure_paths = [
        os.path.join(ARTIFACT_DIR, "ppo_failure_birds_eye.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/failure_demo_birds_eye.mp4")
    ]
    
    print(f"\nRecording Success Episode (Seed {success_seed}) using {camera_name} camera...")
    run_and_record_episode(controller, env_record, success_seed, success_paths, "SAFE - SUCCESS", camera_name)
    
    print(f"\nRecording Failure Episode (Seed {failure_seed}) using {camera_name} camera...")
    run_and_record_episode(controller, env_record, failure_seed, failure_paths, "UNSAFE - FAILURE", camera_name)
    
    env_record.close()
    print("\nFinished recording both birds-eye view episodes successfully!")

if __name__ == "__main__":
    main()
