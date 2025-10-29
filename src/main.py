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
from agents import QLearningAgent
from utils import plot_learning_curve, submit_video
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay-rate", type=float, default=0.9995)
    parser.add_argument("--min-eps", type=float, default=0.05)
    parser.add_argument("--env", type=str, default="CliffWalking-v1")
    parser.add_argument("--num-episodes", type=int, default=80000)
    parser.add_argument("--num-videos", type=int, default=1, help="# of videos to save")
    return parser.parse_args()


        
def train(env, agent : QLearningAgent, num_episodes=100000):
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")
    for episode in pbar:
        state, _ = env.reset()
        terminated, truncated = False, False
        #Added reward tracker for total per episode
        episode_reward = 0  
        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            episode_reward += reward
            state = next_state
        agent.epsilon_decay()
        #Log total reward per episode
        agent.rewards.append(episode_reward)  
    return agent.Q, agent.rewards


def eval_video(env, agent, video_save_path, num_videos):
    agent = copy.deepcopy(agent)
    agent.epsilon = 0.0

    from gymnasium.wrappers import RecordVideo
    venv = RecordVideo(
        env,
        video_folder=video_save_path,
        name_prefix="eval",
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
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Default env
    # env = gym.make(args.env, render_mode="rgb_array")

    # Added custom cliff layout and step limit
    # env = CustomCliffWalkingEnv(shape=(4, 12), render_mode="rgb_array")
    env = CustomCliffWalkingEnv(shape=(4, 12), render_mode="ansi")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    state, info = env.reset()
    
    # Visualize the environment if render_mode is set to "rgb_array"
    #frame = env.render()
    #plt.imshow(frame)
    #plt.axis("off")
    #plt.show()

    # Visualize the environment if render_mode is set to "ansi"
    print(env.render())

    agent = QLearningAgent(
        env,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        decay_rate=args.decay_rate,
        min_eps=args.min_eps
    )
    Q, rewards = train(env, agent, num_episodes=args.num_episodes)
    
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    eval_video(env, agent, video_dir, num_videos=args.num_videos)
    submit_video(video_dir)
    plot_learning_curve(rewards, os.path.join(output_dir, "plot.png"))
    
    shutil.rmtree(video_dir)
    
if __name__ == "__main__":
    main()

