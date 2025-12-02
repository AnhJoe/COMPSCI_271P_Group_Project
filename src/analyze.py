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

# Bootstrap test for difference in decay rates
def bootstrap_decay_diff(q_windows, q_falls, s_windows, s_falls, n_boot=3000):
    boot_diffs = []
    n = len(q_windows)

    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        qw = q_windows[idx]
        qw_f = q_falls[idx]
        sw = s_windows[idx]
        sw_f = s_falls[idx]

        try:
            bq = curve_fit(expo, qw, qw_f, p0=(100, 0.01), maxfev=5000)[0][1]
            bs = curve_fit(expo, sw, sw_f, p0=(100, 0.01), maxfev=5000)[0][1]
            boot_diffs.append(bq - bs)
        except:
            continue  # skip failed fits

    boot_diffs = np.array(boot_diffs)
    lower = np.percentile(boot_diffs, 2.5)
    upper = np.percentile(boot_diffs, 97.5)
    mean = np.mean(boot_diffs)

    return mean, lower, upper, boot_diffs

# Bootstrap confidence interval for variance
def bootstrap_var_diff(a, b, n_boot=3000):
    boot_diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), n, replace=True)
        a_samp = a[idx]
        b_samp = b[idx]
        boot_diffs.append(np.var(a_samp) - np.var(b_samp))
    boot_diffs = np.array(boot_diffs)
    lower = np.percentile(boot_diffs, 2.5)
    upper = np.percentile(boot_diffs, 97.5)
    return np.mean(boot_diffs), lower, upper, boot_diffs


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

    # Convert index to a numeric episode axis
    q_windows = q_grouped.index.values
    s_windows = s_grouped.index.values

    # Restore episode_window_end into grouped DataFrame
    q_grouped["episode_window_end"] = q_df.groupby("episode_window_start")["episode_window_end"].mean()
    s_grouped["episode_window_end"] = s_df.groupby("episode_window_start")["episode_window_end"].mean()
    
    # Extract cliff fall counts and rewards
    q_falls  = q_grouped["cliff_fall_count"].values       
    s_falls  = s_grouped["cliff_fall_count"].values       
    
    # Average reward per 100 episodes
    q_reward = q_grouped["avg_reward_100_eps"].values     
    s_reward = s_grouped["avg_reward_100_eps"].values                      

    # HYPOTHESIS TESTS:

    # H1: Overall across all episodes, does SARSA have fewer cliff falls than Q-Learning?
    # Null: Q has equal or fewer falls than S
    # Extract window-level averages
    q_all = q_grouped["cliff_fall_count"].values
    s_all = s_grouped["cliff_fall_count"].values
    # Ensure equal length (paired test requirement)
    min_len = min(len(q_all), len(s_all))
    q_all = q_all[:min_len]
    s_all = s_all[:min_len]
    # Paired Wilcoxon signed-rank test (Q > S => S has fewer falls)
    stat1, p1 = wilcoxon(s_all, q_all, alternative="less")
    H1_result = p1 < 0.05
    H1_diff_mean, H1_ci_low, H1_ci_high, H1_boot_samples = bootstrap_mean_diff(s_all, q_all)


    # H2: During early learning (first 500 episodes), is SARSA safer early than Q-learning? 
    # Null: SARSA does not have fewer falls than Q in early episodes
    # Identify early windows
    q_early = q_grouped[q_grouped["episode_window_end"] <= 500]["cliff_fall_count"].values
    s_early = s_grouped[s_grouped["episode_window_end"] <= 500]["cliff_fall_count"].values
    # Ensure equal length (paired test requirement)
    min_len = min(len(q_early), len(s_early))
    q_early = q_early[:min_len]
    s_early = s_early[:min_len]
    # Paired Wilcoxon signed-rank test (S > Q => Q has fewer falls)
    stat2, p2 = wilcoxon(s_early, q_early, alternative="less")
    H2_result = p2 < 0.05
    early_diff_mean, early_diff_low, early_diff_high, early_diff_samples = bootstrap_mean_diff(s_early, q_early)


    # H3: Does Q-learning exhibit a faster exponential decay in cliff fall rate than SARSA?
    # Null: Q-learning does not have a higher decay rate than SARSA
    # Fit exponential decay to cliff fall counts
    popt_q, _ = curve_fit(expo, q_windows, q_falls, p0=(100, 0.01), bounds=([0,0], [500,1.0]), maxfev=10000)
    popt_s, _ = curve_fit(expo, s_windows, s_falls, p0=(100, 0.01), bounds=([0,0], [500,1.0]), maxfev=10000)
    # Compare decay rates; positive difference means Q decays faster
    a_q, b_q = popt_q
    a_s, b_s = popt_s
    decay_diff = b_q - b_s
    # Bootstrap CIs for decay rate
    H3_diff_mean, H3_ci_low, H3_ci_high, H3_boot_samples = bootstrap_decay_diff(q_windows, q_falls, s_windows, s_falls)
    # Q-Learning improves faster if decay_diff > 0 and CI lower bound > 0
    H3_result = (decay_diff > 0) and (H3_ci_low > 0)


    #  H4: At the end of training (last 1,000 episodes), is Q-learning more stable (lower variance in reward) than SARSA?
    # Null: Q-learning does not have lower variance than SARSA
    # Compute variance in last K windows
    K = 10
    q_last = q_reward[-K:]
    s_last = s_reward[-K:]
    # Remove length mismatch
    min_len = min(len(q_last), len(s_last))
    q_last = q_last[:min_len]
    s_last = s_last[:min_len]
    # Fligner-Killeen two-sided test for equal variances
    # Test for H0: variances are equal vs H1: variances are different
    _, p_H4 = fligner(q_last, s_last)
    # Q more stable if variance(Q) < variance(S) and p-value < 0.05
    H4_var_diff_mean, H4_ci_low, H4_ci_high, H4_boot_samples = bootstrap_var_diff(q_last, s_last)
    # Compute variances
    q_var_last = np.var(q_last)
    s_var_last = np.var(s_last)
    var_diff = q_var_last - s_var_last
    H4_result = (q_var_last < s_var_last) and (p_H4 < 0.05)

    # H5: In the last 1,000 episodes, does Q-learning achieve higher average rewards than SARSA?
    # Null: Q-learning does not achieve higher average rewards than SARSA
    # Reuse q_last and s_last from H4
    # Reward difference (Q − SARSA)
    reward_diff = q_last - s_last
    obs_reward_diff = np.mean(reward_diff)
    _, p5 = wilcoxon(q_last, s_last, alternative="greater")
    H5_diff_mean, H5_ci_low, H5_ci_high, H5_boot_samples = bootstrap_mean_diff(q_last, s_last)
    H5_result = (obs_reward_diff > 0) and (p5 < 0.05)


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
    plt.title(f"Exponential Decay Comparison - {prefix}")
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
        f.write("## 1. Overall, SARSA will have less cliff falls than Q (H1)\n")
        f.write("### Hypothesis:\n")
        f.write("- **H0:** Overall, SARSA does not have less cliff falls than Q (SARSA >= Q)\n")
        f.write("- **H1:** Overall, SARSA has less cliff falls than Q (SARSA < Q)\n\n")
        f.write(f"- Mean cliff-fall difference (SARSA - Q): {H1_diff_mean:.3f}\n")
        f.write(f"- 95% CI: [{H1_ci_low:.3f}, {H1_ci_high:.3f}]\n")
        f.write(f"- Wilcoxon p-value: {p1:.5f}\n")
        if H1_result:
            f.write("**H1 Conclusion: Supported - SARSA experiences fewer cliff falls overall.**\n")
        else:
            f.write("**H1 Conclusion: Not Supported - No significant overall safety advantage for SARSA.**\n")
        
        # H2: Early learning comparison
        f.write("\n---\n")
        f.write("## 2. In early learning episodes (first 500 episodes), SARSA will have less cliff fall counts than Q (H2)\n")
        f.write("### Hypothesis:\n")
        f.write("- **H0:** In early episodes, SARSA does not have less cliff fall counts than Q (SARSA >= Q)\n")
        f.write("- **H1:** In early episodes, SARSA have less cliff fall counts than Q (SARSA < Q)\n\n")
        f.write(f"- Mean early cliff-fall difference (SARSA - Q): {early_diff_mean:.3f}\n")
        f.write(f"- 95% CI: [{early_diff_low:.3f}, {early_diff_high:.3f}]\n")
        f.write(f"- Wilcoxon p-value: {p2:.5f}\n")
        if H2_result:
            f.write("**H2 Conclusion: Supported - SARSA is safer during early learning.**\n")
        else:
            f.write("**H2 Conclusion: Not Supported - No significant early-learning safety advantage for SARSA.**\n")
        
        # H3: Exponential decay analysis
        f.write("\n---\n")
        f.write("## 3. Does Q learn faster than SARSA?\n")
        f.write("### Hypothesis:\n")
        f.write("- **H0:** Q does not improve faster than SARSA (b_Q <= b_S)\n")
        f.write("- **H1:** Q improves faster (b_Q > b_S)\n\n")
        f.write(f"- Observed decay rate difference (b_Q - b_S): {decay_diff:.5f}\n")
        f.write(f"- Bootstrap mean difference: {H3_diff_mean:.5f}\n")
        f.write(f"- 95% CI for decay difference: [{H3_ci_low:.5f}, {H3_ci_high:.5f}]\n\n")
        if H3_result:
            f.write("**H3 Conclusion: Supported - Q reduces cliff falls at a faster exponential rate.**\n")
        else:
            f.write("**H3 Conclusion: Not Supported - No significant evidence that Q improves faster.**\n")

        # H4: Q-Learning is more stable at the end of training than SARSA
        f.write("---\n")
        f.write("## 4. Is Q more stable at the end of training than SARSA?\n")
        f.write("### Hypothesis:\n")
        f.write("- **H0:** Q does not have lower reward variance than SARSA (Var(Q) >= Var(S))\n")
        f.write("- **H1:** Q has lower reward variance (more stable policy) (Var(Q) < Var(S))\n\n")
        f.write(f"- Variance difference (Q - SARSA): {var_diff:.5f}\n")
        f.write(f"- Bootstrap mean difference: {H4_var_diff_mean:.5f}\n")
        f.write(f"- 95% CI for variance difference: [{H4_ci_low:.5f}, {H4_ci_high:.5f}]\n\n")
        f.write(f"- Fligner-Killeen p-value (two-sided variance test): {p_H4:.5f}\n\n")
        if H4_result:
            f.write("**H4 Conclusion: Supported - Q exhibits significantly lower reward variance and is more stable late in training.**\n")
        else:
            f.write("**H4 Conclusion: Not Supported - No significant evidence that Q is more stable.**\n")

        # H5: Q-Learning finds the optimal path or achieves a higher reward near the end of training than SARSA
        f.write("---\n")
        f.write(f"## 5. Reward Optimality (Higher Reward = Less Negative Value) for last 1,000 episodes\n")
        f.write("### Hypothesis:\n")
        f.write("- **H0:** Q does not achieve higher long-run reward than SARSA (Q <= S)\n")
        f.write("- **H1:** Q achieves higher long-run reward (Q > S)\n\n")
        f.write(f"- Mean reward difference (Q - SARSA): {H5_diff_mean:.3f}\n")
        f.write(f"- 95% CI: [{H5_ci_low:.3f}, {H5_ci_high:.3f}]\n")
        f.write(f"- Wilcoxon p-value: {p5:.5f}\n\n")
        if H5_result:
            f.write("**H5 Conclusion: Supported - Q achieves significantly higher long-run reward.**\n")
        else:
            f.write("**H5 Conclusion: Not Supported - No significant evidence that Q-learning leads to higher long-run reward.**\n")

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
