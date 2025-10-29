import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip
import os


def plot_learning_curve(rewards, save_path, algo_name=""):
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.ylim(-120, 0)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100-episode window)")
    
    title = f"{algo_name.upper()} on CliffWalking" if algo_name else "Learning Curve"
    plt.title(title)

    dirname = os.path.dirname(save_path)
    basename = f"{algo_name}_learning_curve.png" if algo_name else "learning_curve.png"
    out_path = os.path.join(dirname, basename)

    plt.savefig(out_path)
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
