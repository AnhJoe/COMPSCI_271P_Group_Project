import matplotlib.pyplot as plt
import numpy as np
import os
import moviepy as mpy

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
    clip = mpy.VideoFileClip(input_path)
    if max_duration:
        clip = clip.subclip(0, max_duration)
    clip.write_gif(output_path, fps=fps)
    clip.close()


def generate_side_by_side(layout_dir):
    """
    Finds the Q-Learning and SARSA GIFs (whatever their filenames are),
    then stacks them side-by-side into comparison.mp4 and comparison.gif.
    """

    import glob

    # Search for gif inside each algo directory (no matter the filename)
    q_gif_list = glob.glob(os.path.join(layout_dir, "Q-Learning", "*.gif"))
    s_gif_list = glob.glob(os.path.join(layout_dir, "SARSA", "*.gif"))

    if not q_gif_list or not s_gif_list:
        print(f"Cannot generate comparison. GIF missing in: {layout_dir}")
        return

    q_gif = q_gif_list[0]  # first gif found
    s_gif = s_gif_list[0]

    print(f"  Found GIFs:")
    print(f"  Q-Learning: {q_gif}")
    print(f"  SARSA     : {s_gif}")

    # Load GIFs
    q = mpy.VideoFileClip(q_gif)
    s = mpy.VideoFileClip(s_gif)

    # Stack them horizontally
    comparison = mpy.clips_array([[q, s]])

    # Save as MP4 and GIF
    mp4_out = os.path.join(layout_dir, "comparison.mp4")
    gif_out = os.path.join(layout_dir, "comparison.gif")

    comparison.write_videofile(mp4_out, fps=30)
    comparison.write_gif(gif_out, fps=10)

    print(f"Side-by-side comparison created!")
    print(f"{mp4_out}")
    print(f"{gif_out}")
