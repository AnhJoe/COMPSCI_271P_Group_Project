import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load CSVs

Q_PATH = "../data/Q-Learning/Q-Learning_metrics.csv"
S_PATH = "../data/sarsa/SARSA_metrics.csv"

q_df = pd.read_csv(Q_PATH)
s_df = pd.read_csv(S_PATH)

# Clean column names (fix hidden whitespace/bom)
q_df.columns = q_df.columns.str.strip()
s_df.columns = s_df.columns.str.strip()

print("Q columns:", q_df.columns.tolist())
print("S columns:", s_df.columns.tolist())

print("\n Loaded CSVs successfully")
print(f"Q-learning windows: {len(q_df)}")
print(f"SARSA windows: {len(s_df)}")

# Extract vectors for analysis

q_falls = q_df["cliff_fall_count"].values
s_falls = s_df["cliff_fall_count"].values

q_reward = q_df["avg_reward_100_eps"].values
s_reward = s_df["avg_reward_100_eps"].values

q_windows = q_df["episode_window_start"].values
s_windows = s_df["episode_window_start"].values

# H1: Q-Learning has lower mean cliff-fall rate

print("\n H1: Mean cliff-falls comparison (Q < SARSA)")

q_mean = np.mean(q_falls)
s_mean = np.mean(s_falls)

stat, p_value = mannwhitneyu(q_falls, s_falls, alternative="less")

print(f"Q mean falls: {q_mean:.2f}")
print(f"S mean falls: {s_mean:.2f}")
print(f"p-value: {p_value:.5f}")

if p_value < 0.05:
    print(" Reject H0 — evidence that Q-Learning has LOWER fall rate.")
else:
    print(" Fail to reject H0 — no significant difference.\n")

# H2: Early learning (first 500 episodes)

print("\n H2: Early learning (first 500 episodes)")

q_early = q_df[q_df["episode_window_end"] <= 500]["cliff_fall_count"].values
s_early = s_df[s_df["episode_window_end"] <= 500]["cliff_fall_count"].values

stat2, p_value2 = mannwhitneyu(q_early, s_early, alternative="less")

print(f"Q early mean: {np.mean(q_early):.2f}")
print(f"S early mean: {np.mean(s_early):.2f}")
print(f"p-value: {p_value2:.5f}")

if p_value2 < 0.05:
    print(" Reject H0 — Q-Learning learns faster early on.")
else:
    print(" Fail to reject H0 — no early-learning difference.\n")

# H3: Fit exponential decay
#   falls(t) = a * exp(-b * t)

print("\n H3: Exponential decay fitting")

def expo(t, a, b):
    return a * np.exp(-b * t)

# Fit Q-learning
popt_q, _ = curve_fit(expo, q_windows, q_falls, p0=(100, 0.01))
a_q, b_q = popt_q

# Fit SARSA
popt_s, _ = curve_fit(expo, s_windows, s_falls, p0=(100, 0.01))
a_s, b_s = popt_s

print(f"Q-Learning decay rate b: {b_q:.4f}")
print(f"SARSA decay rate b: {b_s:.4f}")

if b_q > b_s:
    print(" Q-Learning improves faster (higher decay rate b).")
else:
    print(" SARSA improves faster or equal.\n")

# Plot exponential fits
plt.figure(figsize=(8,5))
plt.scatter(q_windows, q_falls, label="Q-Learning falls", alpha=0.6)
plt.scatter(s_windows, s_falls, label="SARSA falls", alpha=0.6)
t = np.linspace(1, max(max(q_windows), max(s_windows)), 200)
plt.plot(t, expo(t, *popt_q), label="Q-Learning fit")
plt.plot(t, expo(t, *popt_s), label="SARSA fit")
plt.xlabel("Episode Window")
plt.ylabel("Cliff Falls per 100")
plt.title("Exponential Decay Fit Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("exp_decay_fit.png")
plt.close()

# H4: Reward variance decreases over time

print("\n H4: Reward stability (variance decreasing)")

q_var_first = np.var(q_reward[:3])
q_var_last = np.var(q_reward[-3:])

s_var_first = np.var(s_reward[:3])
s_var_last = np.var(s_reward[-3:])

print(f"Q-Learning var first 3 windows: {q_var_first:.2f}")
print(f"Q-Learning var last 3 windows:  {q_var_last:.2f}")

print(f"SARSA var first 3 windows: {s_var_first:.2f}")
print(f"SARSA var last 3 windows:  {s_var_last:.2f}")

if q_var_last < q_var_first:
    print(" Q-Learning reward stability improved.")

if s_var_last < s_var_first:
    print(" SARSA reward stability improved.")

print("\nAnalysis Complete\n")
