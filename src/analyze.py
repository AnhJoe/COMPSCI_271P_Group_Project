import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu, fligner, wilcoxon
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

from utils import (
    plot_avg_reward_per_100,
    plot_cliff_fall_rate
)

# Exponential decay function for curve fitting
def expo(t, a, b):
    return a * np.exp(-b * t)

# Bootstrap CI of early improvement difference
def bootstrap_mean_diff(a, b, n_boot=3000, ci=95):
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        diffs.append(a[idx] - b[idx])  
    diffs = np.array(diffs)
    lower = np.percentile(diffs, (100-ci)/2)
    upper = np.percentile(diffs, 100 - (100-ci)/2)
    return np.mean(diffs), lower, upper, diffs

# Bootstrap confidence interval for exponential decay rate b
def bootstrap_decay_ci(x, y, n_boot=3000, ci=95):
    boot_b = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        x_samp = x[idx]
        y_samp = y[idx]
        try:
            popt, _ = curve_fit(
            expo, x_samp, y_samp,
            p0=(100, 0.01),
            # Bounds to reduce overflows: a ∈ [0, 500], b ∈ [0, 1]
            bounds=([0, 0], [500, 1.0]),   
            maxfev=10000
            )
            boot_b.append(popt[1])  
        except:
            pass  
    lower = np.percentile(boot_b, (100-ci)/2)
    upper = np.percentile(boot_b, 100 - (100-ci)/2)
    return np.mean(boot_b), lower, upper, boot_b

# Bootstrap test for difference in decay rates
def bootstrap_decay_diff(x_q, y_q, x_s, y_s, n_boot=3000):
    diffs = []
    n_q = len(x_q)
    n_s = len(x_s)
    for _ in range(n_boot):
        idx_q = np.random.choice(np.arange(n_q), size=n_q, replace=True)
        idx_s = np.random.choice(np.arange(n_s), size=n_s, replace=True)
        try:
            bq = curve_fit(expo, x_q[idx_q], y_q[idx_q], p0=(100, 0.01))[0][1]
            bs = curve_fit(expo, x_s[idx_s], y_s[idx_s], p0=(100, 0.01))[0][1]
            diffs.append(bq - bs)
        except:
            pass
    return np.array(diffs)

# Bootstrap confidence interval for variance
def bootstrap_variance_ci(data, n_boot=5000, ci=95):
    boot_samples = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_samples.append(np.var(sample, ddof=1))
    lower = np.percentile(boot_samples, (100-ci)/2)
    upper = np.percentile(boot_samples, 100 - (100-ci)/2)
    return lower, upper


# Main analysis function
def run_analysis(q_csv_path, s_csv_path, output_dir, prefix="baseline"):
    print(f"\n Running analysis for {prefix}")

    # Load data
    q_df = pd.read_csv(q_csv_path)
    s_df = pd.read_csv(s_csv_path)

    # Clean column names
    q_df.columns = q_df.columns.str.strip()
    s_df.columns = s_df.columns.str.strip()

    # Extract metrics and average across 50 runs before hypothesis tests 
    # Group by episode window start i.e., every 100 episodes like 1-100, 101-200, ...
    q_grouped = q_df.groupby("episode_window_start").mean(numeric_only=True)
    s_grouped = s_df.groupby("episode_window_start").mean(numeric_only=True)
    # Restore episode_window_end into grouped DataFrame
    q_grouped["episode_window_end"] = q_df.groupby("episode_window_start")["episode_window_end"].mean()
    s_grouped["episode_window_end"] = s_df.groupby("episode_window_start")["episode_window_end"].mean()
    # Extract cliff fall counts and rewards
    q_falls  = q_grouped["cliff_fall_count"].values       
    s_falls  = s_grouped["cliff_fall_count"].values       
    # Average reward per 100 episodes
    q_reward = q_grouped["avg_reward_100_eps"].values     
    s_reward = s_grouped["avg_reward_100_eps"].values     
    # Episode window indices
    q_windows = q_grouped.index.values                   
    s_windows = s_grouped.index.values                  

    # HYPOTHESIS TESTS:
    # H1: Overall across all episodes, does Q-learning have fewer cliff falls than SARSA?
    # Extract window-level averages (already grouped)
    q_all = q_grouped["cliff_fall_count"].values
    s_all = s_grouped["cliff_fall_count"].values
    # Ensure equal length (they should be)
    min_len = min(len(q_all), len(s_all))
    q_all = q_all[:min_len]
    s_all = s_all[:min_len]
    # Paired Wilcoxon signed-rank test (Q < S => Q has fewer falls)
    stat1, p1 = wilcoxon(q_all, s_all, alternative="less")
    H1_result = p1 < 0.05
    H1_diff_mean, H1_ci_low, H1_ci_high, H1_boot_samples = bootstrap_mean_diff(q_all, s_all)


    # H2: During early learning (first 500 episodes), does Q-learning reduce falls faster than SARSA? 
    # Identify early windows
    q_early = q_grouped[q_grouped["episode_window_end"] <= 500]["cliff_fall_count"].values
    s_early = s_grouped[s_grouped["episode_window_end"] <= 500]["cliff_fall_count"].values
    # Ensure equal length (paired test requirement)
    min_len = min(len(q_early), len(s_early))
    q_early = q_early[:min_len]
    s_early = s_early[:min_len]
    # Paired Wilcoxon signed-rank test (Q < S means Q learns faster)
    stat2, p2 = wilcoxon(q_early, s_early, alternative="less")
    H2_result = p2 < 0.05
    early_diff_mean, early_diff_low, early_diff_high, early_diff_samples = bootstrap_mean_diff(q_early, s_early)


    # H3: Does Q-learning exhibit a faster exponential decay in cliff fall rate than SARSA?
    # Fit exponential decay to cliff fall counts
    popt_q, _ = curve_fit(expo, q_windows, q_falls, p0=(100, 0.01))
    popt_s, _ = curve_fit(expo, s_windows, s_falls, p0=(100, 0.01))
    # Compare decay rates
    a_q, b_q = popt_q
    a_s, b_s = popt_s
    # Bootstrap CIs for decay rate
    b_q_mean, b_q_low, b_q_high, b_q_samples = bootstrap_decay_ci(q_windows, q_falls)
    b_s_mean, b_s_low, b_s_high, b_s_samples = bootstrap_decay_ci(s_windows, s_falls)
    # Bootstrap test for b_q > b_s
    boot_diffs = bootstrap_decay_diff(q_windows, q_falls, s_windows, s_falls)
    p_H3 = np.mean(boot_diffs <= 0)
    H3_result = p_H3 < 0.05 and (b_q_mean > b_s_mean)


    #  H4: At the end of training, is Q-learning more stable (lower variance in reward) than SARSA?
    # Compute variance in last K windows
    K = 10
    q_last = q_reward[-K:]
    s_last = s_reward[-K:]
    # Fligner-Killeen test for homogeneity of variances
    stat_H4, p_H4 = fligner(q_last, s_last)
    # Interpret as Q-learning is more stable if its variance is LOWER
    q_var_end = np.var(q_last)
    s_var_end = np.var(s_last)
    H4_result = (q_var_end < s_var_end) and (p_H4 < 0.05)
    # Bootstrap CIs for variances
    q_ci_low, q_ci_high = bootstrap_variance_ci(q_last)
    s_ci_low, s_ci_high = bootstrap_variance_ci(s_last)


    # PLOTS:
    # Plot averaged metrics
    avg_output = os.path.join(output_dir, "averaged_plots")
    os.makedirs(avg_output, exist_ok=True)

    # Averaged cliff fall rate per 100 episodes
    q_avg_cliff_path = os.path.join(avg_output, f"{prefix}_Q_avg_cliff.png")
    s_avg_cliff_path = os.path.join(avg_output, f"{prefix}_SARSA_avg_cliff.png")
    plot_cliff_fall_rate(q_grouped["cliff_fall_count"].values, q_avg_cliff_path, f"{prefix} Q-Learning (avg)")
    plot_cliff_fall_rate(s_grouped["cliff_fall_count"].values, s_avg_cliff_path, f"{prefix} SARSA (avg)")

    # Averaged reward per 100 episodes
    q_avg_reward_path = os.path.join(avg_output, f"{prefix}_Q_avg_reward.png")
    s_avg_reward_path = os.path.join(avg_output, f"{prefix}_SARSA_avg_reward.png")
    plot_avg_reward_per_100(q_grouped["avg_reward_100_eps"].values, q_avg_reward_path, f"{prefix} Q-Learning (avg)")
    plot_avg_reward_per_100(s_grouped["avg_reward_100_eps"].values, s_avg_reward_path, f"{prefix} SARSA (avg)")

    # Exponential decay fit plot
    t = np.linspace(min(q_windows), max(q_windows), 300)
    plt.figure(figsize=(9, 6))
    plt.scatter(q_windows, q_falls, label="Q-Learning Falls (avg)", alpha=0.5, color="purple")
    plt.scatter(s_windows, s_falls, label="SARSA Falls (avg)", alpha=0.5, color="blue")
    plt.plot(t, expo(t, a_q, b_q), color="red", linewidth=3, label=f"Q Fit (b={b_q:.4f})")
    plt.plot(t, expo(t, a_s, b_s), color="orange", linewidth=3, label=f"SARSA Fit (b={b_s:.4f})")
    plt.xlabel("Episode Window")
    plt.ylabel("Cliff Falls per 100 Episodes")
    plt.title(f"Exponential Decay Comparison – {prefix}")
    plt.legend()
    plt.tight_layout()
    decay_plot_path = os.path.join(output_dir, f"{prefix}_H3_decay_fit.png")
    plt.savefig(decay_plot_path)
    plt.close()

    # Markdown report generation in same output dir
    report_dir = output_dir
    os.makedirs(report_dir, exist_ok=True)
    # Generate timestamped report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"{prefix}_analysis_report_{timestamp}.md")
    
    # Write report
    with open(report_path, "w") as f:

        f.write(f"# Analysis Report - {prefix.capitalize()}\n")
        f.write(f"Generated on: **{timestamp}**\n\n")
        # H1: Overall cliff-fall comparison 
        f.write("\n---\n")
        f.write("## 1. Overall Cliff-Fall Comparison (H1)\n")
        f.write(f"- Mean cliff-fall difference (Q - SARSA): {H1_diff_mean:.3f}\n")
        f.write(f"- 95% CI: [{H1_ci_low:.3f}, {H1_ci_high:.3f}]\n")
        f.write(f"- Wilcoxon p-value: {p1:.5f}\n")
        f.write(f"- **H1 Conclusion**: {'Supported - Q-learning has fewer falls overall.' if H1_result else 'Not supported - No significant overall cliff-fall reduction for Q-learning.'}\n\n")

        # H2: Early learning comparison
        f.write("\n---\n")
        f.write("## 2. Early Learning Analysis (H2)\n")
        f.write(f"- Mean early cliff-fall difference (Q - SARSA): {early_diff_mean:.3f}\n")
        f.write(f"- 95% CI: [{early_diff_low:.3f}, {early_diff_high:.3f}]\n")
        f.write(f"- Wilcoxon p-value: {p2:.5f}\n")
        f.write(f"- **H2 Conclusion**: {'Supported - Q improves faster early.' if H2_result else 'Not supported - No significant early-learning advantage.'}\n\n")

        # H3: Exponential decay analysis
        f.write("---\n")
        f.write("## 3. Exponential Decay Analysis (H3)\n")
        f.write("### Decay Parameter Estimates\n")
        f.write(f"- Q-learning decay rate **b_q = {b_q:.4f}**\n")
        f.write(f"  - Bootstrap mean = {b_q_mean:.4f}\n")
        f.write(f"  - 95% CI = [{b_q_low:.4f}, {b_q_high:.4f}]\n\n")
        f.write(f"- SARSA decay rate **b_s = {b_s:.4f}**\n")
        f.write(f"  - Bootstrap mean = {b_s_mean:.4f}\n")
        f.write(f"  - 95% CI = [{b_s_low:.4f}, {b_s_high:.4f}]\n\n")
        f.write("### Statistical Test\n")
        f.write(f"- Bootstrapped one-sided test b_q > b_s\n")
        f.write(f"- p-value = {p_H3:.5f}\n")
        f.write(f"- **H3 Conclusion**: {'Supported - Q-learning decays significantly faster.' if H3_result else 'Not supported - No significant difference.'}\n\n")

        # H4: Q-Learning is more stable at the end of training than SARSA
        f.write("---\n")
        f.write(f"## 4. Reward Stability for last {K} windows\n")
        f.write(f"- Q-learning variance (last {K} windows): {q_var_end:.4f}\n")
        f.write(f"  - 95% CI: [{q_ci_low:.4f}, {q_ci_high:.4f}]\n")
        f.write(f"- SARSA variance (last {K} windows): {s_var_end:.4f}\n")
        f.write(f"  - 95% CI: [{s_ci_low:.4f}, {s_ci_high:.4f}]\n\n")
        f.write(f"- Fligner-Killeen p-value: {p_H4:.5f}\n")
        f.write(f"- **H4 Conclusion**: {'Q-learning is significantly more stable.' if H4_result else 'No significant evidence that Q-learning is more stable.'}\n")


    print(f"\nMarkdown report saved: {report_path}\n")
    print(f"Analysis complete for {prefix}")

    return {
        "H1_Q_less_than_S": H1_result,
        "H2_Q_learns_faster": H2_result,
        "H3_exp_decay_Q_faster": H3_result,
        "H4_Q_more_stable": H4_result,
        "report_path": report_path,
        "plots_output_dir": avg_output,
    }
