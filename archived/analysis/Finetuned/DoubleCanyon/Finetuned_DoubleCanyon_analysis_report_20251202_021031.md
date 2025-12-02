# Analysis Report - Finetuned_doublecanyon
Generated on: **20251202_021031**


---
## 1. Overall, SARSA will have less cliff falls than Q (H1)
### Hypothesis:
- **H0:** Overall, SARSA does not have less cliff falls than Q (SARSA >= Q)
- **H1:** Overall, SARSA has less cliff falls than Q (SARSA < Q)

- Mean cliff-fall difference (SARSA - Q): -0.393
- 95% CI: [-2.500, 2.400]
- Wilcoxon p-value: 0.00000
**H1 Conclusion: Supported - SARSA experiences fewer cliff falls overall.**

---
## 2. In early learning episodes (first 500 episodes), SARSA will have less cliff fall counts than Q (H2)
### Hypothesis:
- **H0:** In early episodes, SARSA does not have less cliff fall counts than Q (SARSA >= Q)
- **H1:** In early episodes, SARSA have less cliff fall counts than Q (SARSA < Q)

- Mean early cliff-fall difference (SARSA - Q): 4.553
- 95% CI: [0.000, 14.600]
- Wilcoxon p-value: 1.00000
**H2 Conclusion: Not Supported - No significant early-learning safety advantage for SARSA.**

---
## 3. Does Q learn faster than SARSA?
### Hypothesis:
- **H0:** Q does not improve faster than SARSA (b_Q <= b_S)
- **H1:** Q improves faster (b_Q > b_S)

- Observed decay rate difference (b_Q - b_S): 0.00008
- Bootstrap mean difference: 0.00005
- 95% CI for decay difference: [-0.00015, 0.00019]

**H3 Conclusion: Not Supported - No significant evidence that Q improves faster.**
---
## 4. Is Q more stable at the end of training than SARSA?
### Hypothesis:
- **H0:** Q does not have lower reward variance than SARSA (Var(Q) >= Var(S))
- **H1:** Q has lower reward variance (more stable policy) (Var(Q) < Var(S))

- Variance difference (Q - SARSA): -4.30863
- Bootstrap mean difference: -3.85788
- 95% CI for variance difference: [-8.35575, 0.11054]

- Fligner-Killeen p-value (two-sided variance test): 0.13889

**H4 Conclusion: Not Supported - No significant evidence that Q is more stable.**
---
## 5. Reward Optimality (Higher Reward = Less Negative Value) for last 1,000 episodes
### Hypothesis:
- **H0:** Q does not achieve higher long-run reward than SARSA (Q <= S)
- **H1:** Q achieves higher long-run reward (Q > S)

- Mean reward difference (Q - SARSA): 6.243
- 95% CI: [2.679, 11.425]
- Wilcoxon p-value: 0.00098

**H5 Conclusion: Supported - Q achieves significantly higher long-run reward.**
