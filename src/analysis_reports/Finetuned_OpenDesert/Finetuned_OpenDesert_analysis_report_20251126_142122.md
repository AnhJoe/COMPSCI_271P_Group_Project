# Analysis Report – Finetuned_opendesert
Generated on: **20251126_142122**

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
- a = 120.3525
- b = 0.0005

### SARSA
- a = 122.3122
- b = 0.0017

### Decay Fit Plot
![Exponential Decay Fit](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\OpenDesert\Finetuned_OpenDesert_exp_decay_fit.png)

---
## 3. Averaged Metrics Plots
- Q Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\OpenDesert\averaged_plots\Finetuned_OpenDesert_Q_avg_cliff.png)
- SARSA Avg Cliff Fall: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\OpenDesert\averaged_plots\Finetuned_OpenDesert_SARSA_avg_cliff.png)
- Q Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\OpenDesert\averaged_plots\Finetuned_OpenDesert_Q_avg_reward.png)
- SARSA Avg Reward: ![](D:\CS271P Project Repo\COMPSCI_271P_Group_Project\Archive\Finetuned\OpenDesert\averaged_plots\Finetuned_OpenDesert_SARSA_avg_reward.png)

---
## 4. Interpretation Summary
- **H1**: No significant difference.
- **H2**: No evidence of faster early learning.
- **H3**: Decay rates similar or SARSA faster.
- **H4**: Reward stability improved? Q = False, SARSA = False
