# Analysis Report – Baseline_cliffgauntlet
Generated on: **20251126_140453**

## 1. Hypothesis Test Results
| Hypothesis | Interpretation | p-value | Result |
|-----------|----------------|---------|--------|
| H1: Q has fewer falls overall | Mann–Whitney U | 0.00006 | **Supported** |
| H2: Q learns faster early | First 500 episodes | 0.00000 | **Supported** |
| H3: Faster exponential decay | b(Q) > b (SARSA) | — | **Supported** |
| H4: Reward stability | Variance decreasing | — | Q: **True**, SARSA: **False** |

---
## 2. Exponential Fit Parameters
### Q-Learning
- a = 99.7753
- b = 0.0013

### SARSA
- a = 108.3710
- b = 0.0011

### Decay Fit Plot
![Exponential Decay Fit](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\CliffGauntlet\Baseline_CliffGauntlet_exp_decay_fit.png)

---
## 3. Averaged Metrics Plots
- Q Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\CliffGauntlet\averaged_plots\Baseline_CliffGauntlet_Q_avg_cliff.png)
- SARSA Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\CliffGauntlet\averaged_plots\Baseline_CliffGauntlet_SARSA_avg_cliff.png)
- Q Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\CliffGauntlet\averaged_plots\Baseline_CliffGauntlet_Q_avg_reward.png)
- SARSA Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Baseline\CliffGauntlet\averaged_plots\Baseline_CliffGauntlet_SARSA_avg_reward.png)

---
## 4. Interpretation Summary
- **H1**: Q falls significantly less.
- **H2**: Q shows faster early improvement.
- **H3**: Q decays faster (higher b).
- **H4**: Reward stability improved? Q = True, SARSA = False
