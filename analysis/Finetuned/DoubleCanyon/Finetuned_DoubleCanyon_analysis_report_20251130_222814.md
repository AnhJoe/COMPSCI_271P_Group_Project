# Analysis Report - Finetuned_doublecanyon
Generated on: **20251130_222814**


---
## 1. Overall Cliff-Fall Comparison (H1)
- Mean cliff-fall difference (Q - SARSA): 0.743
- 95% CI: [-17.000, 10.000]
- Wilcoxon p-value: 0.99975
- **H1 Conclusion**: Not supported - No significant overall cliff-fall reduction for Q-learning.


---
## 2. Early Learning Analysis (H2)
- Mean early cliff-fall difference (Q - SARSA): -7.374
- 95% CI: [-24.000, 0.000]
- Wilcoxon p-value: 0.12500
- **H2 Conclusion**: Not supported - No significant early-learning advantage.

---
## 3. Exponential Decay Analysis (H3)
### Decay Parameter Estimates
- Q-learning decay rate **b_q = 0.0016**
  - Bootstrap mean = 0.0017
  - 95% CI = [0.0014, 0.0022]

- SARSA decay rate **b_s = 0.0014**
  - Bootstrap mean = 0.0015
  - 95% CI = [0.0012, 0.0022]

### Statistical Test
- Bootstrapped one-sided test b_q > b_s
- p-value = 0.23800
- **H3 Conclusion**: Not supported - No significant difference.

---
## 4. Reward Stability for last 10 windows
- Q-learning variance (last 10 windows): 8.4374
  - 95% CI: [2.7137, 14.7021]
- SARSA variance (last 10 windows): 54.8504
  - 95% CI: [5.9881, 92.2564]

- Fligner-Killeen p-value: 0.45991
- **H4 Conclusion**: No significant evidence that Q-learning is more stable.
