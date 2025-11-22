# Author Ryozo Masukawa (rmasukaw@uci.edu)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from custom_envs import CustomCliffWalkingEnv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm
import os
import copy
import csv

from agents_Q import QLearningAgent
from agents_SARSA import SarsaAgent
from utils import submit_video, plot_avg_reward_per_100, plot_cliff_and_reward, plot_cliff_fall_rate, plot_state_value_heatmap


def parse_args():
    parser = argparse.ArgumentParser()

    # Shared controls for both algorithms
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("--env", type=str, default="CliffWalking-v1")
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--num-videos", type=int, default=1, help="# of videos to save")
    # CHANGE NUM-RUNS BACK TO 1 WHEN YOU'RE DONE
    parser.add_argument("--num-runs", type=int, default=1, help="How many times to repeat the full experiment")
    
    # BASELINE Q-Learning hyperparameters (DONE)
    parser.add_argument("--qlearning-gamma", dest="QLearning_gamma", type=float, default=0.95)
    parser.add_argument("--qlearning-alpha", dest="QLearning_alpha", type=float, default=0.5)  
    parser.add_argument("--qlearning-epsilon", dest="QLearning_epsilon", type=float, default=1.0)
    parser.add_argument("--qlearning-decay-rate", dest="QLearning_decay_rate", type=float, default=0.999)
    parser.add_argument("--qlearning-min-eps", dest="QLearning_min_eps", type=float, default=0.05)

    # BASELINE SARSA hyperparameters (DONE)
    # parser.add_argument("--sarsa-gamma", dest="SARSA_gamma", type=float, default=0.95)
    # parser.add_argument("--sarsa-alpha", dest="SARSA_alpha", type=float, default=0.3)          
    # parser.add_argument("--sarsa-epsilon", dest="SARSA_epsilon", type=float, default=1.0)
    # parser.add_argument("--sarsa-decay-rate", dest="SARSA_decay_rate", type=float, default=0.999)
    # parser.add_argument("--sarsa-min-eps", dest="SARSA_min_eps", type=float, default=0.05)

    # # TUNED Q-Learning hyperparameters (WORKING)
    # parser.add_argument("--qlearning-gamma", dest="QLearning_gamma", type=float, default=0.95)
    # parser.add_argument("--qlearning-alpha", dest="QLearning_alpha", type=float, default=0.5)  
    # parser.add_argument("--qlearning-epsilon", dest="QLearning_epsilon", type=float, default=1.0)
    # parser.add_argument("--qlearning-decay-rate", dest="QLearning_decay_rate", type=float, default=0.999)
    # parser.add_argument("--qlearning-min-eps", dest="QLearning_min_eps", type=float, default=0.05)

    # TUNED SARSA hyperparameters (DONE)
    parser.add_argument("--sarsa-gamma", dest="SARSA_gamma", type=float, default=0.99)
    parser.add_argument("--sarsa-alpha", dest="SARSA_alpha", type=float, default=0.15)          
    parser.add_argument("--sarsa-epsilon", dest="SARSA_epsilon", type=float, default=1.0)
    parser.add_argument("--sarsa-decay-rate", dest="SARSA_decay_rate", type=float, default=0.997)
    parser.add_argument("--sarsa-min-eps", dest="SARSA_min_eps", type=float, default=0.02)

    return parser.parse_args()



def train(env, agent, num_episodes=100000):
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")

     # Define metrics
    cliff_falls_overall = 0
    cliff_falls_per_100 = []
    avg_reward_per_100 = []

    falls_this_window = 0
    rewards_this_window = []

    # Safe upper bound so SARSA doesn’t get stuck
    MAX_STEPS_PER_EPISODE = 500  

    for episode in pbar:
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0
        steps = 0

        # Detect algorithm type by name
        is_sarsa = hasattr(agent, "returns_next_action")

        if is_sarsa:
            action = agent.get_action(state)

        while not (terminated or truncated):
            
            # Limit steps per episode
            steps += 1
            if steps >= MAX_STEPS_PER_EPISODE:
                truncated = True     
                break
            
            # SARSA loop
            if is_sarsa:
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Count cliff fall
                if reward == -100:
                    cliff_falls_overall += 1
                    falls_this_window += 1

                action = agent.update(state, action, reward, next_state, terminated or truncated)
                state = next_state

            # Q-learning loop
            else:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Count cliff fall
                if reward == -100:
                    cliff_falls_overall += 1
                    falls_this_window += 1

                agent.update(state, action, reward, next_state, terminated or truncated)
                state = next_state

        agent.epsilon_decay()
        # agent.rewards.append(episode_reward)

        # Track 100-episode window stats
        # rewards_this_window.append(episode_reward)

        # if (episode + 1) % 100 == 0:
        #     cliff_falls_per_100.append(falls_this_window)
        #     avg_reward_per_100.append(np.mean(rewards_this_window))

        #     falls_this_window = 0
        #     rewards_this_window = []

    return {
        "Q": agent.Q,
        "episode_rewards": agent.rewards,
        "cliff_falls_overall": cliff_falls_overall,
        "cliff_falls_per_100": cliff_falls_per_100,
        "avg_reward_per_100": avg_reward_per_100
    }

# Generate run_id for primary keys
def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# CSV pipeline to store metrics
def save_metrics_csv(metrics, output_dir, algorithm, run_id):
    csv_path = os.path.join(output_dir, f"{algorithm}_metrics.csv")

    cliff_falls_per_100 = metrics["cliff_falls_per_100"]
    avg_reward_per_100 = metrics["avg_reward_per_100"]

    cumulative_falls = 0

    header = [
        "run_id",
        "algorithm",
        "episode_window_start",
        "episode_window_end",
        "cliff_fall_count",
        "cliff_fall_rate_overall",
        "avg_reward_100_eps"
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

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

    print(f"CSV appended successfully → {csv_path}")
    return csv_path


def eval_video(env, agent, video_save_path, num_videos, algo_name=""):
    MAX_EVAL_STEPS = 500
    steps = 0

    agent = copy.deepcopy(agent)
    agent.epsilon = 0.0
    
    filename_prefix = f"{algo_name}_" if algo_name else ""

    venv = RecordVideo(
        env,
        video_folder=video_save_path,
        name_prefix=f"{filename_prefix}eval",
        episode_trigger=lambda ep: True,
    )
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
    args = parse_args()

    algos = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent
    }
    for run_idx in range(args.num_runs):
        print(f"\n==== Starting Run {run_idx + 1} / {args.num_runs} =====\n")

        # Generate run_id once per run (keeps CSV rows grouped cleanly)
        run_id = generate_run_id()
        for algo_name, AlgoClass in algos.items():
            print(f"\n Training {algo_name.upper()}:\n")

            # Create fresh env
            env = CustomCliffWalkingEnv(render_mode="rgb_array")
            state, info = env.reset()

            # Create unique output directory per algo
            output_dir = os.path.join(args.output, algo_name)
            os.makedirs(output_dir, exist_ok=True)



            # Initialize agents (Q-Learning & SARSA)
            if algo_name == "Q-Learning":
                agent = AlgoClass(
                    env,
                    gamma=args.QLearning_gamma,
                    alpha=args.QLearning_alpha,
                    epsilon=args.QLearning_epsilon,
                    decay_rate=args.QLearning_decay_rate,
                    min_eps=args.QLearning_min_eps,
                )

            elif algo_name == "SARSA":
                agent = AlgoClass(
                    env,
                    gamma=args.SARSA_gamma,
                    alpha=args.SARSA_alpha,
                    epsilon=args.SARSA_epsilon,
                    decay_rate=args.SARSA_decay_rate,
                    min_eps=args.SARSA_min_eps,
                )

            # Train agent
            metrics = train(env, agent, num_episodes=args.num_episodes)
            q_table = metrics["Q"]
            episode_rewards = metrics["episode_rewards"]
            cliff_falls_overall = metrics["cliff_falls_overall"]
            cliff_falls_per_100 = metrics["cliff_falls_per_100"]
            avg_reward_per_100 = metrics["avg_reward_per_100"]

            # Save Q-table
            qtable_path = os.path.join(output_dir, f"{algo_name}_q_table.npy")
            np.save(qtable_path, q_table)

            # Save metrics to CSV
            # run_id = generate_run_id()
            # csv_path = save_metrics_csv(
            #     metrics=metrics,
            #     output_dir=output_dir,
            #     algorithm=algo_name,
            #     run_id=run_id
            # )

            # Cliff fall plot
            cliff_plot_path = os.path.join(output_dir, f"{algo_name}_cliff_falls.png")
            plot_cliff_fall_rate(cliff_falls_per_100, cliff_plot_path, algo_name)

            # Avg reward per 100 episode plot
            reward100_plot = os.path.join(output_dir, f"{algo_name}_avg_reward_100.png")
            plot_avg_reward_per_100(avg_reward_per_100, reward100_plot, algo_name)

            # Combined plot
            combined_plot = os.path.join(output_dir, f"{algo_name}_cliff_vs_reward.png")
            plot_cliff_and_reward(cliff_falls_per_100, avg_reward_per_100, combined_plot, algo_name)

            # State-value heatmap
            env.unwrapped
            heatmap_path = os.path.join(output_dir, f"{algo_name}_value_heatmap.png")
            plot_state_value_heatmap(q_table, env.rows, env.cols, heatmap_path, algo_name)

            # Create and save evaluation video
            video_dir = os.path.join(output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            eval_video(env, agent, video_dir, num_videos=args.num_videos, algo_name=algo_name)
            submit_video(video_dir)

    print("\n===== All runs completed successfully =====\n")

if __name__ == "__main__":
    main()

