# Author: Updated for layout-based analysis
# NOTE FOR TEAM:

# The training block below is COMMENTED OUT intentionally.

# How to re-enable training:
# 1. Uncomment the entire block
# 2. Ensure the output directory (data/Baseline or data/Finetuned)
#    is correct for your run

# If you ONLY want to run analysis:
# → Keep this block commented.


import gymnasium as gym
from custom_envs import CustomCliffWalkingEnv
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


# ARGUMENTS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=str, default="data")
    parser.add_argument("--layout", type=str,
                        default="CliffGauntlet",
                        choices=["CliffGauntlet", "DoubleCanyon", "OpenDesert"])

    parser.add_argument("--mode", type=str, default="Baseline",
                        choices=["Baseline", "Finetuned"])

    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=1)

    # Q-learning hyperparams
    parser.add_argument("--qlearning-gamma", type=float, default=0.95)
    parser.add_argument("--qlearning-alpha", type=float, default=0.5)
    parser.add_argument("--qlearning-epsilon", type=float, default=1.0)
    parser.add_argument("--qlearning-decay-rate", type=float, default=0.999)
    parser.add_argument("--qlearning-min-eps", type=float, default=0.05)

    # SARSA hyperparams
    parser.add_argument("--sarsa-gamma", type=float, default=0.95)
    parser.add_argument("--sarsa-alpha", type=float, default=0.3)
    parser.add_argument("--sarsa-epsilon", type=float, default=1.0)
    parser.add_argument("--sarsa-decay-rate", type=float, default=0.999)
    parser.add_argument("--sarsa-min-eps", type=float, default=0.05)

    return parser.parse_args()


# TRAINING

def train(env, agent, num_episodes=10000):
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")

    cliff_falls_per_100 = []
    avg_reward_per_100 = []

    falls = 0
    window_rewards = []
    MAX_STEPS = 500

    for episode in pbar:
        state, _ = env.reset()
        reward_sum = 0
        terminated = truncated = False
        steps = 0

        is_sarsa = hasattr(agent, "returns_next_action")
        if is_sarsa:
            action = agent.get_action(state)

        while not (terminated or truncated):
            steps += 1
            if steps >= MAX_STEPS:
                truncated = True
                break

            if is_sarsa:
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                if reward == -100:
                    falls += 1
                action = agent.update(state, action, reward, next_state, terminated)
                state = next_state
            else:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                if reward == -100:
                    falls += 1
                agent.update(state, action, reward, next_state, terminated)
                state = next_state

        agent.epsilon_decay()
        window_rewards.append(reward_sum)

        if (episode + 1) % 100 == 0:
            cliff_falls_per_100.append(falls)
            avg_reward_per_100.append(np.mean(window_rewards))
            falls = 0
            window_rewards = []

    return {
        "Q": agent.Q,
        "cliff_falls_per_100": cliff_falls_per_100,
        "avg_reward_per_100": avg_reward_per_100,
    }


# CSV SAVE

def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_metrics_csv(metrics, output_dir, layout_name, algo_name, run_id):

    filename = f"{layout_name}_{algo_name}_metrics.csv"
    csv_path = os.path.join(output_dir, filename)

    header = [
        "run_id", "algorithm", "episode_window_start", "episode_window_end",
        "cliff_fall_count", "avg_reward_100_eps"
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for idx, falls in enumerate(metrics["cliff_falls_per_100"]):
            window_start = idx * 100 + 1
            window_end = (idx + 1) * 100
            reward_avg = metrics["avg_reward_per_100"][idx]

            writer.writerow([
                run_id,
                algo_name,
                window_start,
                window_end,
                falls,
                reward_avg
            ])

    return csv_path


# MAIN

def main():
    args = parse_args()

    # Folder such as:
    # data/Baseline/
    # data/Finetuned/
    src_dir = os.path.dirname(os.path.abspath(__file__))              # /src
    project_root = os.path.dirname(src_dir)                           # project root
    data_root = os.path.join(project_root, "data")                    # /data
    archive_root = os.path.join(project_root, "Archive")              # /Archive

    mode_dir = os.path.join(data_root, args.mode)                     # Baseline or Finetuned
    os.makedirs(mode_dir, exist_ok=True)

    layout_name = args.layout

    run_id = generate_run_id()

    algorithms = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent
    }

    for run_idx in range(args.num_runs):
        print(f"\n===== RUN {run_idx + 1}/{args.num_runs} =====\n")
        run_id = generate_run_id()
        
        # Uncomment this part for training algos

        # csv_paths = {}

        
        # # TRAIN BOTH ALGORITHMS
        
        # for algo_name, AlgoClass in algorithms.items():

        #     print(f"\nTraining {algo_name} on {layout_name} ({args.mode})...")

        #     # Load the custom env (ASCII map already defined)
        #     env = CustomCliffWalkingEnv(render_mode="rgb_array")
        #     env.reset()

        #     if algo_name == "Q-Learning":
        #         agent = AlgoClass(
        #             env,
        #             gamma=args.qlearning_gamma,
        #             alpha=args.qlearning_alpha,
        #             epsilon=args.qlearning_epsilon,
        #             decay_rate=args.qlearning_decay_rate,
        #             min_eps=args.qlearning_min_eps
        #         )
        #     else:
        #         agent = AlgoClass(
        #             env,
        #             gamma=args.sarsa_gamma,
        #             alpha=args.sarsa_alpha,
        #             epsilon=args.sarsa_epsilon,
        #             decay_rate=args.sarsa_decay_rate,
        #             min_eps=args.sarsa_min_eps
        #         )

        #     metrics = train(env, agent, num_episodes=args.num_episodes)

        #     # Save CSV using *layout-based* file names
        #     csv_path = save_metrics_csv(
        #         metrics=metrics,
        #         output_dir=mode_dir,
        #         layout_name=layout_name,
        #         algo_name=algo_name,
        #         run_id=run_id
        #     )
        #     csv_paths[algo_name] = csv_path

        
        # RUN ANALYSIS
        
        csv_paths = {
        "Q-Learning": os.path.join(
            mode_dir, f"{layout_name}_Q-Learning_metrics.csv"
        ),
        "SARSA": os.path.join(
            mode_dir, f"{layout_name}_SARSA_metrics.csv"
        ),
    } # Comment out when training
        
        print("Looking for:")
        print(" →", csv_paths["Q-Learning"])
        print(" →", csv_paths["SARSA"])
        
        # Analysis output folder
        
        analysis_out = os.path.join(archive_root, args.mode, layout_name)
        os.makedirs(analysis_out, exist_ok=True)

        print(f"\nAnalysis Output Folder: {analysis_out}")

        # Run Analysis
        
        print("\nRunning analysis...")
        run_analysis(
        q_csv_path=csv_paths["Q-Learning"],
        s_csv_path=csv_paths["SARSA"],
        output_dir=analysis_out,
        prefix=f"{args.mode}_{layout_name}"
    )

    print("\nAll analysis finished.\n")


if __name__ == "__main__":
    main()
