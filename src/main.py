import gymnasium as gym
from custom_envs import CustomCliffWalkingEnv
from custom_envs import load_layout
from gymnasium.wrappers import RecordVideo

from agents_Q import QLearningAgent
from agents_SARSA import SarsaAgent

from utils import (
    submit_video,
    plot_avg_reward_per_100,
    plot_cliff_and_reward,
    plot_cliff_fall_rate,
    plot_state_value_heatmap,
)

from analyze import run_analysis

import numpy as np
import argparse
import tqdm
import os
import csv
from datetime import datetime
import copy
import glob
from collections import defaultdict

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    # Shared controls for both algorithms
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("--env", type=str, default="CliffWalking-v1", help="Gym environment name")
    parser.add_argument("--num-episodes", type=int, default=20000, help="# of training episodes per run")
    parser.add_argument("--num-videos", type=int, default=1, help="# of videos to save")
    parser.add_argument("--num-runs", type=int, default=1, help="How many times to repeat the full experiment")
    parser.add_argument( "--analysis-only", action="store_true", help="Run analysis only using existing CSV files, without training or appending new rows")
    parser.add_argument("--train-only", action="store_true", help="Run training only (no analysis)")
    return parser.parse_args()

# Training loop
def train(env, agent, num_episodes=10000):
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")
    # Define metrics
    cliff_falls_overall = 0
    cliff_falls_per_100 = []
    avg_reward_per_100 = []
    falls_this_window = 0
    rewards_this_window = []
    # Safe upper bound so SARSA doesn’t get stuck
    MAX_STEPS_PER_EPISODE = 500  
    # Training loop
    for episode in pbar:
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0
        steps = 0
        # Check if agent is SARSA
        is_sarsa = hasattr(agent, "returns_next_action")
        # Initialize action for SARSA
        if is_sarsa:
            action = agent.get_action(state)
        # Episode loop
        while not (terminated or truncated):
            steps += 1
            # Prevent infinite loops
            if steps >= MAX_STEPS_PER_EPISODE:
                truncated = True
                break     
            # SARSA Loop
            if is_sarsa:
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                # Count cliff fall
                if reward == -100:
                    cliff_falls_overall += 1
                    falls_this_window += 1
                # Get next action
                action = agent.update(state, action, reward, next_state, terminated or truncated)
                state = next_state
            # Q-Learning Loop
            else:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if reward == -100:
                    cliff_falls_overall += 1
                    falls_this_window += 1
                agent.update(state, action, reward, next_state, terminated or truncated)
                state = next_state
        
        # End of episode updates
        agent.epsilon_decay()
        # Record episode reward
        agent.rewards.append(episode_reward)
        # Track 100-episode window stats
        rewards_this_window.append(episode_reward)
        # Every 100 episodes, log metrics
        if (episode + 1) % 100 == 0:
            cliff_falls_per_100.append(falls_this_window)
            avg_reward_per_100.append(np.mean(rewards_this_window))
            falls_this_window = 0
            rewards_this_window = []
    # Return metrics
    return {
        "Q": agent.Q,
        "episode_rewards": agent.rewards,
        "cliff_falls_overall": cliff_falls_overall,
        "cliff_falls_per_100": cliff_falls_per_100,
        "avg_reward_per_100": avg_reward_per_100
    }

# Generate run_id based on timestamp
def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV pipeline to store metrics
def save_metrics_csv(metrics, output_dir, layout_name, algorithm, run_id):
    # Create CSV file path
    csv_path = os.path.join(output_dir, f"{layout_name}_{algorithm}_metrics.csv")
    # Extract metrics
    cliff_falls_per_100 = metrics["cliff_falls_per_100"]
    avg_reward_per_100 = metrics["avg_reward_per_100"]
    # Initialize cumulative falls
    cumulative_falls = 0
    # Write to CSV
    header = [
        "run_id",
        "algorithm",
        "episode_window_start",
        "episode_window_end",
        "cliff_fall_count",
        "cliff_fall_rate_overall",
        "avg_reward_100_eps"
    ]
    # Check if file exists to write header
    file_exists = os.path.exists(csv_path)
    # Append data to CSV
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(header)
        # Write metrics per 100-episode window
        for idx, falls in enumerate(cliff_falls_per_100):
            cumulative_falls += falls
            reward_avg = avg_reward_per_100[idx]
            window_start = idx * 100 + 1
            window_end = (idx + 1) * 100
            row = [
                run_id,
                algorithm,
                window_start,
                window_end,
                falls,
                cumulative_falls,
                reward_avg
            ]
            writer.writerow(row)
    print(f"CSV appended successfully: {csv_path}")
    return csv_path

# Evaluation video generation
def eval_video(env, agent, video_save_path, num_videos, algo_name=""):
    # Max steps per evaluation episode
    MAX_EVAL_STEPS = 500
    # Create evaluation environment with video recording
    steps = 0
    agent = copy.deepcopy(agent)
    agent.epsilon = 0.0
    # Set up video recording
    filename_prefix = f"{algo_name}_" if algo_name else ""
    venv = RecordVideo(
        env.unwrapped,
        video_folder=video_save_path,
        name_prefix=f"{filename_prefix}eval",
        episode_trigger=lambda ep: True,
    )
    # Run evaluation episodes
    for _ in range(num_videos):
        state, _ = venv.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            if steps >= MAX_EVAL_STEPS:
                break
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = venv.step(action)
            steps += 1
    venv.close()  

def main():
    # Parse arguments
    args = parse_args()
    # Safety check for flags
    if args.analysis_only and args.train_only:
        raise ValueError("Cannot use --analysis-only and --train-only at the same time.")
    
    # Define experiment configurations
    all_modes = ["Finetuned"]
    layout_names = ["CliffGauntlet", "DoubleCanyon", "OpenDesert"]
    hyperparameters = {
    "Baseline": {
        "Q-Learning": dict(gamma=0.95, alpha=0.5, epsilon=1.0, decay_rate=0.999, min_eps=0.05),
        "SARSA":      dict(gamma=0.95, alpha=0.5, epsilon=1.0, decay_rate=0.999, min_eps=0.05),
    },

    "Finetuned": {
        "Q-Learning": dict(gamma=0.99, alpha=0.15, epsilon=1.0, decay_rate=0.997, min_eps=0.02),
        "SARSA":      dict(gamma=0.99, alpha=0.15, epsilon=1.0, decay_rate=0.997,  min_eps=0.02),
    }
    }
    # Dictionary to hold all CSV paths
    csv_paths = {}
    # Algorithm mapping
    algos = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent
    }
    # TRAINING LOOP
    if not args.analysis_only:
        # Loop through modes: Baseline and Finetuned
        for mode in all_modes:
            print(f"\n Starting Mode: {mode}\n")
            
            # Loop through layouts: CliffGauntlet, DoubleCanyon, OpenDesert
            for layout_name in layout_names:
                print(f"\n Starting experiments on layout: {layout_name} \n")
                csv_paths[(mode, layout_name)] = {}
                
                # Loop through number of runs
                for run_idx in range(args.num_runs):
                    print(f"\n Starting Run {run_idx + 1} / {args.num_runs}\n")

                    # Generate run_id for each run
                    run_id = generate_run_id()
                    
                    for algo_name, AlgoClass in algos.items():
                        print(f"\n Training {algo_name.upper()} on {layout_name}:\n")

                        # Create fresh env per algo
                        load_layout(layout_name)
                        env = CustomCliffWalkingEnv(render_mode="rgb_array")
                        env.reset()

                        # Create unique output directory per algo
                        output_dir = os.path.join(args.output, mode, layout_name, algo_name)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Initialize agent with hyperparameters
                        params = hyperparameters[mode][algo_name]
                        agent = AlgoClass(
                            env,
                            gamma=params["gamma"],
                            alpha=params["alpha"],
                            epsilon=params["epsilon"],
                            decay_rate=params["decay_rate"],
                            min_eps=params["min_eps"],
                        )

                        # Train agent
                        metrics = train(env, agent, num_episodes=args.num_episodes)
                        
                        # Save metrics to CSV
                        csv_path = save_metrics_csv(
                            metrics=metrics,
                            output_dir=output_dir,
                            layout_name=layout_name,
                            algorithm=algo_name,
                            run_id=run_id
                        )

                        # Store CSV paths for analysis
                        csv_paths[(mode, layout_name)][algo_name] = csv_path
                        
                        # # Create and save evaluation video
                        # video_dir = os.path.join(output_dir, "videos")
                        # os.makedirs(video_dir, exist_ok=True)
                        # eval_video(env, agent, video_dir, num_videos=args.num_videos, algo_name=algo_name)
                        # submit_video(video_dir)
    # ANALYSIS LOOP
    if not args.train_only:
        print("\nStarting Analysis Phase...\n")
        # Dictionary to hold all paths
        csv_paths = defaultdict(dict)
        # Gather CSV files for all modes, layouts, and algorithms
        for mode in all_modes:
            for layout_name in layout_names:
                for algo_name in algos.keys():
                    pattern = os.path.join(args.output, mode, layout_name, algo_name, f"{layout_name}_{algo_name}_metrics.csv")
                    csv_files = sorted(glob.glob(pattern))
                    if csv_files:
                        # Take the first CSV file found
                        csv_paths[(mode, layout_name)][algo_name] = csv_files[0]
                        print(f"Found CSV for {mode} / {layout_name} / {algo_name}: {csv_files[0]}") 
            
        # Run analysis for each mode/layout where both algos exist
        src_dir = os.path.dirname(os.path.abspath(__file__))  
        project_root = os.path.dirname(src_dir)
        # Loop through modes and layouts
        for (mode, layout_name), algo_dict in csv_paths.items(): 
            if "Q-Learning" not in algo_dict or "SARSA" not in algo_dict:
                print(f"Skipping analysis for {mode} / {layout_name}: missing Q-Learning or SARSA CSV.")
                continue
            print(f"\nRunning Analysis for Mode: {mode}, Layout: {layout_name}\n")
            # Create analysis output directory
            analysis_out = os.path.join(project_root, "analysis", mode, layout_name)
            os.makedirs(analysis_out, exist_ok=True)
            # Run analysis
            run_analysis(
                q_csv_path=algo_dict["Q-Learning"],
                s_csv_path=algo_dict["SARSA"],
                output_dir=analysis_out,
                prefix=f"{mode}_{layout_name}",
            )

if __name__ == "__main__":
    main()
