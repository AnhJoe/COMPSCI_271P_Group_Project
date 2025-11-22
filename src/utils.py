import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip
import os


def plot_cliff_and_reward(cliff_falls_per_100, avg_reward_per_100, save_path, algorithm):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(f"{algorithm} – Cliff Falls vs Avg Reward")
    ax1.set_xlabel("100-Episode Windows")

    # Cliff falls (left axis)
    line1 = ax1.plot(
        cliff_falls_per_100,
        color="red",
        marker="x",
        label="Cliff Falls"
    )[0]
    ax1.set_ylabel("Cliff Falls", color="red")
    ax1.tick_params(axis='y', labelcolor='red')

    # Avg reward (right axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        avg_reward_per_100,
        color="blue",
        marker="o",
        label="Avg Reward"
    )[0]
    ax2.set_ylabel("Average Reward", color="blue")
    ax2.tick_params(axis='y', labelcolor='blue')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_state_value_heatmap(q_table, rows, cols, save_path, algorithm):
    state_values = np.max(q_table, axis=1)
    grid_values = state_values.reshape((rows, cols))

    plt.figure(figsize=(8, 6))
    plt.imshow(grid_values, cmap="viridis")
    plt.colorbar(label="State Value")
    plt.title(f"{algorithm} – State Value Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Add legend-like text
    plt.text(
        0.02, 0.02,
        "Colors represent max Q-value per state",
        transform=plt.gca().transAxes,
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4)
    )

    plt.savefig(save_path)
    plt.close()


def plot_avg_reward_per_100(avg_reward_per_100, save_path, algorithm):
    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward_per_100, marker='o', color='green', label="Avg Reward")
    plt.title(f"{algorithm} – Average Reward per 100 Episodes")
    plt.xlabel("100-Episode Windows")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cliff_fall_rate(cliff_falls_per_100, save_path, algorithm):
    plt.figure(figsize=(10, 5))
    plt.plot(cliff_falls_per_100, marker='o', label="Cliff Falls")
    plt.title(f"{algorithm} – Cliff Fall Rate per 100 Episodes")
    plt.xlabel("100-Episode Windows")
    plt.ylabel("Cliff Falls")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def submit_video(video_dir):
    out_dir = os.path.dirname(video_dir)
    
    for path in os.listdir(video_dir):
        if path.endswith(".mp4"):
            input_file = os.path.join(video_dir, path)
            base = os.path.splitext(path)[0] + ".gif"
            output_file = os.path.join(out_dir, base)
            mp4_to_gif(input_file, output_file)


def mp4_to_gif(input_path, output_path, fps=20, max_duration=None):
    clip = VideoFileClip(input_path)
    if max_duration:
        clip = clip.subclip(0, max_duration)
    clip.write_gif(output_path, fps=fps)
    clip.close()
