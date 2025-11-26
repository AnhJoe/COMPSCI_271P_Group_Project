# Analysis Report – Finetuned_doublecanyon
Generated on: **20251126_142058**

## 1. Hypothesis Test Results
| Hypothesis | Interpretation | p-value | Result |
|-----------|----------------|---------|--------|
| H1: Q has fewer falls overall | Mann–Whitney U | 1.00000 | **Not supported** |
| H2: Q learns faster early | First 500 episodes | 1.00000 | **Not supported** |
| H3: Faster exponential decay | b(Q) > b (SARSA) | — | **Not supported** |
| H4: Reward stability | Variance decreasing | — | Q: **False**, SARSA: **False** |

---
## 2. Exponential Fit Parameters
### Q-Learning
- a = 122.4924
- b = 0.0004

### SARSA
- a = 121.8082
- b = 0.0015

### Decay Fit Plot
![Exponential Decay Fit](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\DoubleCanyon\Finetuned_DoubleCanyon_exp_decay_fit.png)

---
## 3. Averaged Metrics Plots
- Q Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\DoubleCanyon\averaged_plots\Finetuned_DoubleCanyon_Q_avg_cliff.png)
- SARSA Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\DoubleCanyon\averaged_plots\Finetuned_DoubleCanyon_SARSA_avg_cliff.png)
- Q Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\DoubleCanyon\averaged_plots\Finetuned_DoubleCanyon_Q_avg_reward.png)
- SARSA Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\DoubleCanyon\averaged_plots\Finetuned_DoubleCanyon_SARSA_avg_reward.png)

---
## 4. Interpretation Summary
- **H1**: No significant difference.
- **H2**: No evidence of faster early learning.
- **H3**: Decay rates similar or SARSA faster.
- **H4**: Reward stability improved? Q = False, SARSA = False
