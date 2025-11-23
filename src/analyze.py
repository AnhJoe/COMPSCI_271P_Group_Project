import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

from utils import (
    plot_avg_reward_per_100,
    plot_cliff_and_reward,
    plot_cliff_fall_rate,
    plot_state_value_heatmap
)

def expo(t, a, b):
    return a * np.exp(-b * t)


def run_analysis(q_csv_path, s_csv_path, output_dir, prefix="baseline"):
    """
    Runs hypothesis tests, produces analysis plots, averaged metrics,
    and generates a Markdown analysis report.
    """

    print(f"\n Running analysis for {prefix}")
    print("Loading CSVs...")

    # Load
    q_df = pd.read_csv(q_csv_path)
    s_df = pd.read_csv(s_csv_path)

    q_df.columns = q_df.columns.str.strip()
    s_df.columns = s_df.columns.str.strip()

    # Extract metrics
    q_falls = q_df["cliff_fall_count"].values
    s_falls = s_df["cliff_fall_count"].values

    q_reward = q_df["avg_reward_100_eps"].values
    s_reward = s_df["avg_reward_100_eps"].values

    q_windows = q_df["episode_window_start"].values
    s_windows = s_df["episode_window_start"].values

    #  H1: Cliff-fall rate
    stat1, p1 = mannwhitneyu(q_falls, s_falls, alternative="less")
    h1_result = p1 < 0.05

    # H2: Early learning (first 500 episodes) 
    q_early = q_df[q_df["episode_window_end"] <= 500]["cliff_fall_count"].values
    s_early = s_df[s_df["episode_window_end"] <= 500]["cliff_fall_count"].values

    stat2, p2 = mannwhitneyu(q_early, s_early, alternative="less")
    h2_result = p2 < 0.05

    # H3: Exponential decay fit
    popt_q, _ = curve_fit(expo, q_windows, q_falls, p0=(100, 0.01))
    popt_s, _ = curve_fit(expo, s_windows, s_falls, p0=(100, 0.01))

    a_q, b_q = popt_q
    a_s, b_s = popt_s
    h3_result = b_q > b_s

    # Exponential decay plot
    t = np.linspace(1, max(max(q_windows), max(s_windows)), 200)

    plt.figure(figsize=(8, 5))
    plt.scatter(q_windows, q_falls, label="Q-Learning falls", alpha=0.6)
    plt.scatter(s_windows, s_falls, label="SARSA falls", alpha=0.6)
    plt.plot(t, expo(t, *popt_q), label="Q-learning fit")
    plt.plot(t, expo(t, *popt_s), label="SARSA fit")
    plt.xlabel("Episode Window")
    plt.ylabel("Cliff Falls per 100 episodes")
    plt.title(f"Exponential Decay Fit ({prefix})")
    plt.legend()
    plt.tight_layout()

    decay_plot_path = os.path.join(output_dir, f"{prefix}_exp_decay_fit.png")
    plt.savefig(decay_plot_path)
    plt.close()

    #  H4: Reward stability 
    q_var_first = np.var(q_reward[:3])
    q_var_last = np.var(q_reward[-3:])
    s_var_first = np.var(s_reward[:3])
    s_var_last = np.var(s_reward[-3:])

    q_stability_improved = q_var_last < q_var_first
    s_stability_improved = s_var_last < s_var_first

    # Average across multiple runs 
    q_avg = q_df.groupby("episode_window_start").mean(numeric_only=True)
    s_avg = s_df.groupby("episode_window_start").mean(numeric_only=True)

    avg_output = os.path.join(output_dir, "averaged_plots")
    os.makedirs(avg_output, exist_ok=True)

    # averaged cliff falls
    q_avg_cliff = os.path.join(avg_output, f"{prefix}_Q_avg_cliff.png")
    s_avg_cliff = os.path.join(avg_output, f"{prefix}_SARSA_avg_cliff.png")
    plot_cliff_fall_rate(q_avg["cliff_fall_count"].values, q_avg_cliff, f"{prefix} Q-Learning (avg)")
    plot_cliff_fall_rate(s_avg["cliff_fall_count"].values, s_avg_cliff, f"{prefix} SARSA (avg)")

    # averaged reward
    q_avg_reward_path = os.path.join(avg_output, f"{prefix}_Q_avg_reward.png")
    s_avg_reward_path = os.path.join(avg_output, f"{prefix}_SARSA_avg_reward.png")
    plot_avg_reward_per_100(q_avg["avg_reward_100_eps"].values, q_avg_reward_path, f"{prefix} Q-Learning (avg)")
    plot_avg_reward_per_100(s_avg["avg_reward_100_eps"].values, s_avg_reward_path, f"{prefix} SARSA (avg)")

    # MARKDOWN REPORT GENERATION

    report_dir = os.path.join("analysis_reports", prefix)
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"{prefix}_analysis_report_{timestamp}.md")

    with open(report_path, "w") as f:

        f.write(f"# Analysis Report – {prefix.capitalize()}\n")
        f.write(f"Generated on: **{timestamp}**\n\n")
        f.write("## 1. Hypothesis Test Results\n")

        f.write("| Hypothesis | Interpretation | Result |\n")
        f.write("|-----------|----------------|--------|\n")
        f.write(f"| H1: Q has fewer falls overall | Mann–Whitney U | **{'Supported' if h1_result else 'Not supported'}** |\n")
        f.write(f"| H2: Q learns faster early | First 500 episodes | **{'Supported' if h2_result else 'Not supported'}** |\n")
        f.write(f"| H3: Faster exponential decay | b(Q) > b (SARSA) | **{'Supported' if h3_result else 'Not supported'}** |\n")
        f.write(f"| H4: Reward stability | Variance decreasing | Q: **{q_stability_improved}**, SARSA: **{s_stability_improved}** |\n")

        f.write("\n---\n")
        f.write("## 2. Exponential Fit Parameters\n")
        f.write("### Q-Learning\n")
        f.write(f"- a = {a_q:.4f}\n")
        f.write(f"- b = {b_q:.4f}\n\n")
        f.write("### SARSA\n")
        f.write(f"- a = {a_s:.4f}\n")
        f.write(f"- b = {b_s:.4f}\n\n")

        f.write("### Decay Fit Plot\n")
        f.write(f"![Exponential Decay Fit]({decay_plot_path})\n\n")

        f.write("---\n")
        f.write("## 3. Averaged Metrics Plots\n")
        f.write(f"- Q Avg Cliff Fall: ![]({q_avg_cliff})\n")
        f.write(f"- SARSA Avg Cliff Fall: ![]({s_avg_cliff})\n")
        f.write(f"- Q Avg Reward: ![]({q_avg_reward_path})\n")
        f.write(f"- SARSA Avg Reward: ![]({s_avg_reward_path})\n\n")

        f.write("---\n")
        f.write("## 4. Interpretation Summary\n")
        f.write(f"- **H1**: {'Q falls significantly less.' if h1_result else 'No significant difference.'}\n")
        f.write(f"- **H2**: {'Q shows faster early improvement.' if h2_result else 'No evidence of faster early learning.'}\n")
        f.write(f"- **H3**: {'Q decays faster (higher b).' if h3_result else 'Decay rates similar or SARSA faster.'}\n")
        f.write(f"- **H4**: Reward stability improved? Q = {q_stability_improved}, SARSA = {s_stability_improved}\n")

    print(f"\nMarkdown report saved → {report_path}\n")
    print(f"Analysis complete for {prefix}")

    return {
        "H1_Q_less_than_S": h1_result,
        "H2_Q_learns_faster": h2_result,
        "H3_exp_decay_Q_faster": h3_result,
        "H4_Q_reward_stability": q_stability_improved,
        "H4_S_reward_stability": s_stability_improved,
        "report_path": report_path,
        "plots_output_dir": avg_output,
    }
