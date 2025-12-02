# Analysis Report - Baseline_opendesert
Generated on: **20251202_021022**


---
## 1. Overall, SARSA will have less cliff falls than Q (H1)
### Hypothesis:
- **H0:** Overall, SARSA does not have less cliff falls than Q (SARSA >= Q)
- **H1:** Overall, SARSA has less cliff falls than Q (SARSA < Q)

- Mean cliff-fall difference (SARSA - Q): -4.261
- 95% CI: [-10.700, 19.500]
- Wilcoxon p-value: 0.00000
**H1 Conclusion: Supported - SARSA experiences fewer cliff falls overall.**

---
## 2. In early learning episodes (first 500 episodes), SARSA will have less cliff fall counts than Q (H2)
### Hypothesis:
- **H0:** In early episodes, SARSA does not have less cliff fall counts than Q (SARSA >= Q)
- **H1:** In early episodes, SARSA have less cliff fall counts than Q (SARSA < Q)

- Mean early cliff-fall difference (SARSA - Q): 0.021
- 95% CI: [0.000, 0.100]
- Wilcoxon p-value: 1.00000
**H2 Conclusion: Not Supported - No significant early-learning safety advantage for SARSA.**

---
## 3. Does Q learn faster than SARSA?
### Hypothesis:
- **H0:** Q does not improve faster than SARSA (b_Q <= b_S)
- **H1:** Q improves faster (b_Q > b_S)

- Observed decay rate difference (b_Q - b_S): -0.00009
- Bootstrap mean difference: -0.00009
- 95% CI for decay difference: [-0.00012, -0.00007]

**H3 Conclusion: Not Supported - No significant evidence that Q improves faster.**
---
## 4. Is Q more stable at the end of training than SARSA?
### Hypothesis:
- **H0:** Q does not have lower reward variance than SARSA (Var(Q) >= Var(S))
- **H1:** Q has lower reward variance (more stable policy) (Var(Q) < Var(S))

- Variance difference (Q - SARSA): -7.64119
- Bootstrap mean difference: -6.96282
- 95% CI for variance difference: [-12.13189, -2.46218]

- Fligner-Killeen p-value (two-sided variance test): 0.01025

**H4 Conclusion: Supported - Q exhibits significantly lower reward variance and is more stable late in training.**
---
## 5. Reward Optimality (Higher Reward = Less Negative Value) for last 1,000 episodes
### Hypothesis:
- **H0:** Q does not achieve higher long-run reward than SARSA (Q <= S)
- **H1:** Q achieves higher long-run reward (Q > S)

- Mean reward difference (Q - SARSA): 424.923
- 95% CI: [417.134, 429.145]
- Wilcoxon p-value: 0.00098

**H5 Conclusion: Supported - Q achieves significantly higher long-run reward.**
