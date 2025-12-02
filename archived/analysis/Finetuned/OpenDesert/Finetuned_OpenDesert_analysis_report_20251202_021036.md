# Analysis Report - Finetuned_opendesert
Generated on: **20251202_021036**


---
## 1. Overall, SARSA will have less cliff falls than Q (H1)
### Hypothesis:
- **H0:** Overall, SARSA does not have less cliff falls than Q (SARSA >= Q)
- **H1:** Overall, SARSA has less cliff falls than Q (SARSA < Q)

- Mean cliff-fall difference (SARSA - Q): -0.548
- 95% CI: [-2.000, 2.300]
- Wilcoxon p-value: 0.00000
**H1 Conclusion: Supported - SARSA experiences fewer cliff falls overall.**

---
## 2. In early learning episodes (first 500 episodes), SARSA will have less cliff fall counts than Q (H2)
### Hypothesis:
- **H0:** In early episodes, SARSA does not have less cliff fall counts than Q (SARSA >= Q)
- **H1:** In early episodes, SARSA have less cliff fall counts than Q (SARSA < Q)

- Mean early cliff-fall difference (SARSA - Q): 0.142
- 95% CI: [-1.000, 1.100]
- Wilcoxon p-value: 0.56250
**H2 Conclusion: Not Supported - No significant early-learning safety advantage for SARSA.**

---
## 3. Does Q learn faster than SARSA?
### Hypothesis:
- **H0:** Q does not improve faster than SARSA (b_Q <= b_S)
- **H1:** Q improves faster (b_Q > b_S)

- Observed decay rate difference (b_Q - b_S): 0.00007
- Bootstrap mean difference: 0.00009
- 95% CI for decay difference: [0.00002, 0.00025]

**H3 Conclusion: Supported - Q reduces cliff falls at a faster exponential rate.**
---
## 4. Is Q more stable at the end of training than SARSA?
### Hypothesis:
- **H0:** Q does not have lower reward variance than SARSA (Var(Q) >= Var(S))
- **H1:** Q has lower reward variance (more stable policy) (Var(Q) < Var(S))

- Variance difference (Q - SARSA): -5.80853
- Bootstrap mean difference: -5.17798
- 95% CI for variance difference: [-11.67078, -0.41212]

- Fligner-Killeen p-value (two-sided variance test): 0.01456

**H4 Conclusion: Supported - Q exhibits significantly lower reward variance and is more stable late in training.**
---
## 5. Reward Optimality (Higher Reward = Less Negative Value) for last 1,000 episodes
### Hypothesis:
- **H0:** Q does not achieve higher long-run reward than SARSA (Q <= S)
- **H1:** Q achieves higher long-run reward (Q > S)

- Mean reward difference (Q - SARSA): 2.739
- 95% CI: [0.552, 9.768]
- Wilcoxon p-value: 0.00098

**H5 Conclusion: Supported - Q achieves significantly higher long-run reward.**
