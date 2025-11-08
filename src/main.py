# Author Ryozo Masukawa (rmasukaw@uci.edu)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from custom_envs import CustomCliffWalkingEnv

import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm
import os
import copy
from agents_Q import QLearningAgent
from agents_SARSA import SarsaAgent
from utils import plot_learning_curve, submit_video, generate_side_by_side

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay-rate", type=float, default=0.9995)
    parser.add_argument("--min-eps", type=float, default=0.05)
    parser.add_argument("--env", type=str, default="CliffWalking-v1")
    parser.add_argument("--num-episodes", type=int, default=100000)
    parser.add_argument("--num-videos", type=int, default=1, help="# of videos to save")
    return parser.parse_args()



def train(env, agent, num_episodes=70000):
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")

    for episode in pbar:
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0

        # Detect algorithm type by class name
        is_sarsa = hasattr(agent, "returns_next_action")

        if is_sarsa:
            # SARSA: get initial action
            # Force exploratory behavior early
            # agent.epsilon = max(agent.epsilon, 0.3)
            action = agent.get_action(state)

        while not (terminated or truncated):
            if is_sarsa:
                # -------------- SARSA LOOP --------------
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                action = agent.update(state, action, reward, next_state,
                                      terminated or truncated)
                state = next_state

            else:
                # --------- Q-LEARNING LOOP --------------
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                agent.update(state, action, reward, next_state,
                             terminated or truncated)
                state = next_state

        
        agent.rewards.append(episode_reward)
        
        agent.epsilon_decay()

    return agent.Q, agent.rewards




def eval_video(env, agent, video_save_path, num_videos, algo_name=""):
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
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = venv.step(action)
            steps += 1
    venv.close()  



def main():
    args = parse_args()
    args.num_episodes = 50000   # override to 50k episodes

    layouts = ["default", "gauntlet", "loop"]
    algos = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent
    }

    for layout in layouts:
        print(f"\n==============================")
        print(f"Running Layout: {layout}")
        print(f"==============================")

        layout_dir = os.path.join(args.output, layout)
        os.makedirs(layout_dir, exist_ok=True)

        trained_agents = {}

        for algo_name, AlgoClass in algos.items():
            print(f"\n Training {algo_name} on {layout}...")

            env = CustomCliffWalkingEnv(layout=layout, render_mode="rgb_array")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

            output_dir = os.path.join(layout_dir, algo_name)
            os.makedirs(output_dir, exist_ok=True)

            agent = AlgoClass(
                env,
                gamma=args.gamma,
                alpha=args.alpha,
                epsilon=args.epsilon,
                decay_rate=args.decay_rate,
                min_eps=args.min_eps
            )

            Q, rewards = train(env, agent, num_episodes=args.num_episodes)

            trained_agents[algo_name] = agent

            plot_learning_curve(
                rewards,
                os.path.join(output_dir, f"{algo_name}.png"),
                algo_name
            )

            video_dir = os.path.join(output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

            eval_video(env, agent, video_dir, num_videos=args.num_videos, algo_name=algo_name)
            submit_video(video_dir)

        # when both are trained -> generate side-by-side video and plot
        generate_side_by_side(layout_dir)

    print("\n Finished 3 layouts with both agents.")
    
if __name__ == "__main__":
    main()
        


