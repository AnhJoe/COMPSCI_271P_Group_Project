# Analysis Report - Baseline_doublecanyon
Generated on: **20251202_021015**


---
## 1. Overall, SARSA will have less cliff falls than Q (H1)
### Hypothesis:
- **H0:** Overall, SARSA does not have less cliff falls than Q (SARSA >= Q)
- **H1:** Overall, SARSA has less cliff falls than Q (SARSA < Q)

- Mean cliff-fall difference (SARSA - Q): 25.975
- 95% CI: [0.000, 33.900]
- Wilcoxon p-value: 1.00000
**H1 Conclusion: Not Supported - No significant overall safety advantage for SARSA.**

---
## 2. In early learning episodes (first 500 episodes), SARSA will have less cliff fall counts than Q (H2)
### Hypothesis:
- **H0:** In early episodes, SARSA does not have less cliff fall counts than Q (SARSA >= Q)
- **H1:** In early episodes, SARSA have less cliff fall counts than Q (SARSA < Q)

- Mean early cliff-fall difference (SARSA - Q): -0.040
- 95% CI: [-0.100, 0.000]
- Wilcoxon p-value: 0.25000
**H2 Conclusion: Not Supported - No significant early-learning safety advantage for SARSA.**

---
## 3. Does Q learn faster than SARSA?
### Hypothesis:
- **H0:** Q does not improve faster than SARSA (b_Q <= b_S)
- **H1:** Q improves faster (b_Q > b_S)

- Observed decay rate difference (b_Q - b_S): 0.00005
- Bootstrap mean difference: 0.00005
- 95% CI for decay difference: [0.00004, 0.00007]

**H3 Conclusion: Supported - Q reduces cliff falls at a faster exponential rate.**
---
## 4. Is Q more stable at the end of training than SARSA?
### Hypothesis:
- **H0:** Q does not have lower reward variance than SARSA (Var(Q) >= Var(S))
- **H1:** Q has lower reward variance (more stable policy) (Var(Q) < Var(S))

- Variance difference (Q - SARSA): -157.94795
- Bootstrap mean difference: -141.27454
- 95% CI for variance difference: [-319.36728, -10.02980]

- Fligner-Killeen p-value (two-sided variance test): 0.00674

**H4 Conclusion: Supported - Q exhibits significantly lower reward variance and is more stable late in training.**
---
## 5. Reward Optimality (Higher Reward = Less Negative Value) for last 1,000 episodes
### Hypothesis:
- **H0:** Q does not achieve higher long-run reward than SARSA (Q <= S)
- **H1:** Q achieves higher long-run reward (Q > S)

- Mean reward difference (Q - SARSA): 295.787
- 95% CI: [275.029, 327.631]
- Wilcoxon p-value: 0.00098

**H5 Conclusion: Supported - Q achieves significantly higher long-run reward.**
