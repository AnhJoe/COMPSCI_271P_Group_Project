# Analysis Report - Finetuned_cliffgauntlet
Generated on: **20251130_222752**


---
## 1. Overall Cliff-Fall Comparison (H1)
- Mean cliff-fall difference (Q - SARSA): 0.535
- 95% CI: [-3.000, 4.000]
- Wilcoxon p-value: 1.00000
- **H1 Conclusion**: Not supported - No significant overall cliff-fall reduction for Q-learning.


---
## 2. Early Learning Analysis (H2)
- Mean early cliff-fall difference (Q - SARSA): -1.440
- 95% CI: [-13.000, 12.000]
- Wilcoxon p-value: 0.43750
- **H2 Conclusion**: Not supported - No significant early-learning advantage.

---
## 3. Exponential Decay Analysis (H3)
### Decay Parameter Estimates
- Q-learning decay rate **b_q = 0.0039**
  - Bootstrap mean = 0.0042
  - 95% CI = [0.0033, 0.0054]

- SARSA decay rate **b_s = 0.0040**
  - Bootstrap mean = 0.0049
  - 95% CI = [0.0023, 0.0106]

### Statistical Test
- Bootstrapped one-sided test b_q > b_s
- p-value = 0.58833
- **H3 Conclusion**: Not supported - No significant difference.

---
## 4. Reward Stability for last 10 windows
- Q-learning variance (last 10 windows): 0.4413
  - 95% CI: [0.1546, 0.7922]
- SARSA variance (last 10 windows): 0.0110
  - 95% CI: [0.0036, 0.0211]

- Fligner-Killeen p-value: 0.32755
- **H4 Conclusion**: No significant evidence that Q-learning is more stable.
