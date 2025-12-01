# Analysis Report - Finetuned_opendesert
Generated on: **20251130_222841**


---
## 1. Overall Cliff-Fall Comparison (H1)
- Mean cliff-fall difference (Q - SARSA): 0.502
- 95% CI: [-8.000, 8.000]
- Wilcoxon p-value: 0.97748
- **H1 Conclusion**: Not supported - No significant overall cliff-fall reduction for Q-learning.


---
## 2. Early Learning Analysis (H2)
- Mean early cliff-fall difference (Q - SARSA): 1.569
- 95% CI: [-2.000, 8.000]
- Wilcoxon p-value: 0.87500
- **H2 Conclusion**: Not supported - No significant early-learning advantage.

---
## 3. Exponential Decay Analysis (H3)
### Decay Parameter Estimates
- Q-learning decay rate **b_q = 0.0018**
  - Bootstrap mean = 0.0020
  - 95% CI = [0.0015, 0.0032]

- SARSA decay rate **b_s = 0.0017**
  - Bootstrap mean = 0.0019
  - 95% CI = [0.0014, 0.0029]

### Statistical Test
- Bootstrapped one-sided test b_q > b_s
- p-value = 0.42867
- **H3 Conclusion**: Not supported - No significant difference.

---
## 4. Reward Stability for last 10 windows
- Q-learning variance (last 10 windows): 4.4084
  - 95% CI: [1.6138, 7.3098]
- SARSA variance (last 10 windows): 96.0404
  - 95% CI: [6.2354, 171.1957]

- Fligner-Killeen p-value: 0.04746
- **H4 Conclusion**: Q-learning is significantly more stable.
