# Analysis Report – Baseline_opendesert
Generated on: **20251126_140710**

## 1. Hypothesis Test Results
| Hypothesis | Interpretation | p-value | Result |
|-----------|----------------|---------|--------|
| H1: Q has fewer falls overall | Mann–Whitney U | 1.00000 | **Not supported** |
| H2: Q learns faster early | First 500 episodes | 0.04218 | **Supported** |
| H3: Faster exponential decay | b(Q) > b (SARSA) | — | **Not supported** |
| H4: Reward stability | Variance decreasing | — | Q: **False**, SARSA: **False** |

---
## 2. Exponential Fit Parameters
### Q-Learning
- a = 109.7531
- b = 0.0005

### SARSA
- a = 127.4397
- b = 0.0009

### Decay Fit Plot
![Exponential Decay Fit](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\OpenDesert\Baseline_OpenDesert_exp_decay_fit.png)

---
## 3. Averaged Metrics Plots
- Q Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\OpenDesert\averaged_plots\Baseline_OpenDesert_Q_avg_cliff.png)
- SARSA Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\OpenDesert\averaged_plots\Baseline_OpenDesert_SARSA_avg_cliff.png)
- Q Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\OpenDesert\averaged_plots\Baseline_OpenDesert_Q_avg_reward.png)
- SARSA Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\OpenDesert\averaged_plots\Baseline_OpenDesert_SARSA_avg_reward.png)

---
## 4. Interpretation Summary
- **H1**: No significant difference.
- **H2**: Q shows faster early improvement.
- **H3**: Decay rates similar or SARSA faster.
- **H4**: Reward stability improved? Q = False, SARSA = False
