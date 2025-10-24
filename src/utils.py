import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip
import os


def plot_learning_curve(rewards, save_path):
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    # Trim the y-axis to focus on relevant reward range
    plt.ylim(-120, 0)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100-episode window)")
    # Change the plot title if necessary
    plt.title("Q-Learning on CliffWalking-v1")
    plt.savefig(save_path)
    
    
def submit_video(video_dir):
    out_dir = os.path.dirname(video_dir)
    
    for path in os.listdir(video_dir):
        if path.endswith(".mp4"):
            mp4_to_gif(os.path.join(video_dir, path),out_dir)


def mp4_to_gif(input_path, submission_dir, fps=20, max_duration=None): #Changed FPS to 20, Optional: trim clip duration if max_duration is set (seconds)
    clip = VideoFileClip(input_path)
    if max_duration:
        clip = clip.subclip(0, max_duration)   
    gif_path = os.path.basename(input_path.replace(".mp4", ".gif"))
    gif_path = os.path.join(submission_dir, gif_path)
    clip.write_gif(gif_path, fps=fps)


