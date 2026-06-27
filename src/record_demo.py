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

def make_env_rgb(
    normalize_obs: bool = True,
    smooth_actions: bool = False,
    action_alpha: float = 0.8,
    frame_stack: int = 1,
    obs_stats_path: str = None,
):
    # Use rgb_array render_mode
    env = safety_gymnasium.make("SafetyRacecarButton2-v0", render_mode="rgb_array")
    
    from environment_setup import CostTrackingWrapper, FrameStackWrapper, ObsNormWrapper, ActionSmoothingWrapper
    env = CostTrackingWrapper(env)
    
    if frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)
        
    if normalize_obs:
        env = ObsNormWrapper(env, update=False)
        if obs_stats_path and os.path.exists(obs_stats_path):
            env.load_stats(obs_stats_path)
            print(f"[Record] Loaded observation stats from {obs_stats_path}")
        else:
            print("[Warning] No observation stats loaded!")
            
    if smooth_actions:
        env = ActionSmoothingWrapper(env, alpha=action_alpha)
        
    return env

def overlay_info(frame, step, reward, cost, seed, status_text):
    # Convert numpy array to PIL Image
    img = Image.fromarray(frame)
    # Upscale to 512x512
    img = img.resize((512, 512), Image.Resampling.NEAREST)
    
    draw = ImageDraw.Draw(img)
    # Draw a semi-transparent black rectangle at the top
    draw.rectangle([0, 0, 512, 60], fill=(0, 0, 0))
    
    # Text sizes and positions
    text_line1 = f"Seed: {seed} | Step: {step} | Controller: PPO-Lagrangian"
    text_line2 = f"Cumulative Reward: {reward:.3f} | Cost: {cost:.1f} ({status_text})"
    
    # Use default PIL font
    draw.text((10, 10), text_line1, fill=(255, 255, 255))
    draw.text((10, 32), text_line2, fill=(255, 255, 255) if cost <= 25.0 else (255, 100, 100))
    
    return np.array(img)

def run_evaluation_only(controller, env, seed):
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = controller.act(obs)
        obs, reward, cost, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        total_cost += cost
        steps += 1
        
    return total_reward, total_cost, steps

def run_and_record_episode(controller, env, seed, video_paths, status_text):
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)
    
    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    frames = []
    
    # Initial frame
    frame = env.render()
    if frame is not None:
        frames.append(overlay_info(frame, steps, total_reward, total_cost, seed, status_text))
        
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = controller.act(obs)
        obs, reward, cost, terminated, truncated, step_info = env.step(action)
        
        total_reward += reward
        total_cost += cost
        steps += 1
        
        frame = env.render()
        if frame is not None:
            frames.append(overlay_info(frame, steps, total_reward, total_cost, seed, status_text))
            
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
    
    # We will first evaluate seeds 0 to 19 (non-rendering, fast) to find best success & worst failure
    print("Evaluating seeds 0 to 19...")
    env_eval = make_env_rgb(normalize_obs=True, obs_stats_path=obs_stats_path)
    
    seeds = list(range(20))
    results = []
    
    for seed in tqdm(seeds, desc="Evaluating"):
        rew, cost, steps = run_evaluation_only(controller, env_eval, seed)
        results.append({
            "seed": seed,
            "reward": rew,
            "cost": cost,
            "steps": steps
        })
        
    env_eval.close()
    
    # Print results summary
    print("\n--- Evaluation Results ---")
    print(f"{'Seed':<6} | {'Reward':<10} | {'Cost':<10} | {'Status':<10}")
    print("-" * 45)
    for r in results:
        status = "SAFE" if r["cost"] <= 25.0 else "UNSAFE"
        print(f"{r['seed']:<6} | {r['reward']:<10.3f} | {r['cost']:<10.1f} | {status:<10}")
        
    # Find representative success: Cost <= 25, highest reward
    safe_runs = [r for r in results if r["cost"] <= 25.0]
    if safe_runs:
        success_run = max(safe_runs, key=lambda x: x["reward"])
    else:
        # fallback: lowest cost run
        success_run = min(results, key=lambda x: x["cost"])
        
    # Find representative failure: Cost > 25, highest cost
    unsafe_runs = [r for r in results if r["cost"] > 25.0]
    if unsafe_runs:
        failure_run = max(unsafe_runs, key=lambda x: x["cost"])
    else:
        # fallback: highest cost run
        failure_run = max(results, key=lambda x: x["cost"])
        
    print(f"\nSelected Success Episode: Seed {success_run['seed']} (Reward: {success_run['reward']:.3f}, Cost: {success_run['cost']:.1f})")
    print(f"Selected Failure Episode: Seed {failure_run['seed']} (Reward: {failure_run['reward']:.3f}, Cost: {failure_run['cost']:.1f})")
    
    # Now record both episodes
    env_record = make_env_rgb(normalize_obs=True, obs_stats_path=obs_stats_path)
    
    # Paths to save
    success_paths = [
        os.path.join(ARTIFACT_DIR, "ppo_success.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/success_demo.mp4")
    ]
    
    failure_paths = [
        os.path.join(ARTIFACT_DIR, "ppo_failure.mp4"),
        os.path.join(PROJECT_DIR, "runs/ppo_lagrangian/ppo_model2/failure_demo.mp4")
    ]
    
    print("\nRecording Success Episode...")
    run_and_record_episode(controller, env_record, success_run["seed"], success_paths, "SAFE - SUCCESS")
    
    print("\nRecording Failure Episode...")
    run_and_record_episode(controller, env_record, failure_run["seed"], failure_paths, "UNSAFE - FAILURE")
    
    env_record.close()
    print("\nFinished recording both episodes successfully!")

if __name__ == "__main__":
    main()
