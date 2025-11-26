# Analysis Report – Finetuned_cliffgauntlet
Generated on: **20251126_142036**

## 1. Hypothesis Test Results
| Hypothesis | Interpretation | p-value | Result |
|-----------|----------------|---------|--------|
| H1: Q has fewer falls overall | Mann–Whitney U | 1.00000 | **Not supported** |
| H2: Q learns faster early | First 500 episodes | 1.00000 | **Not supported** |
| H3: Faster exponential decay | b(Q) > b (SARSA) | — | **Not supported** |
| H4: Reward stability | Variance decreasing | — | Q: **True**, SARSA: **True** |

---
## 2. Exponential Fit Parameters
### Q-Learning
- a = 101.9719
- b = 0.0009

### SARSA
- a = 106.6914
- b = 0.0041

### Decay Fit Plot
![Exponential Decay Fit](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\CliffGauntlet\Finetuned_CliffGauntlet_exp_decay_fit.png)

---
## 3. Averaged Metrics Plots
- Q Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\CliffGauntlet\averaged_plots\Finetuned_CliffGauntlet_Q_avg_cliff.png)
- SARSA Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\CliffGauntlet\averaged_plots\Finetuned_CliffGauntlet_SARSA_avg_cliff.png)
- Q Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\CliffGauntlet\averaged_plots\Finetuned_CliffGauntlet_Q_avg_reward.png)
- SARSA Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\CliffGauntlet\averaged_plots\Finetuned_CliffGauntlet_SARSA_avg_reward.png)

---
## 4. Interpretation Summary
- **H1**: No significant difference.
- **H2**: No evidence of faster early learning.
- **H3**: Decay rates similar or SARSA faster.
- **H4**: Reward stability improved? Q = True, SARSA = True
