## OVERVIEW ##
This repository implements a complete experimental framework to compare Q-Learning and SARSA across multiple custom CliffWalking layouts using Gymnasium.
It includes:
- Custom environments
- Training loops
- Finetuned and baseline hyperparameter modes
- CSV metric logging for rewards and cliff fall counts
- Evaluation video recording for .gif
- Full statistical analysis pipeline with five hypothesis tests
- Bootstrap confidence intervals
- Markdown reports & plots

## FOLDER STRUCTURE ##
project/

src/
- main.py               # Orchestration
- agents_Q.py           # Q-Learning
- agents_SARSA.py       # SARSA
- custom_envs.py        # Custom ASCII-map envs
- analyze.py            # Statistical tests, plots, reports
- utils.py              # Plotting utilities, video exports

data/
- {Baseline|Finetuned}/{Layout}/{Algo}/...csv # Output path for --train-only

analysis/
- {Baseline|Finetuned}/{Layout}/...md + plots # Output path for --analysis-only

archive/
- ...                   # Snapshot of final results

requirements.txt

## GIT ##
git clone

pip install -r requirements.txt

## ARGUMENTS USAGE ##
python src/main.py [OPTIONS]
| Argument          | Values                                       | Default | Meaning                                          |
| ----------------- | -------------------------------------------- | ------- | ------------------------------------------------ |
| `--num-episodes`  | int                                          | 20000   | Episodes per run                                 |
| `--num-runs`      | int                                          | 1       | How many independent runs (each appended to CSV) |
| `--algo`          | qlearning, sarsa, both                       | both    | Select algorithm(s)                              |
| `--mode`          | baseline, finetuned, both                    | both    | Hyperparameter mode                              |
| `--layout`        | cliffgauntlet, doublecanyon, opendesert, all | all     | Environment layout                               |
| `--analysis-only` | flag                                         | false   | Skip training; analyze existing CSVs             |
| `--train-only`    | flag                                         | false   | Skip analysis                                    |
| `--gamma`         | float                                        | None    | Override mode gamma                              |
| `--alpha`         | float                                        | None    | Override mode α                                  |
| `--epsilon`       | float                                        | None    | Override initial ε                               |
| `--decay-rate`    | float                                        | None    | Override epsilon decay                           |
| `--min-eps`       | float                                        | None    | Override min epsilon                             |


## STATISTICAL TESTS ##
| Hypothesis | Comparison                        | Test                         | Why                     |
| ---------- | --------------------------------- | ---------------------------- | ----------------------- |
| **H1**     | SARSA safer overall               | Wilcoxon (paired, one-sided) | Window-level Q vs SARSA |
| **H2**     | SARSA safer early (first 500 eps) | Wilcoxon one-sided           | Early windows ≤ 500     |
| **H3**     | Q-learning learns faster          | Exp decay + bootstrap        | Tests decay rate b      |
| **H4**     | Q-learning more stable            | Fligner two-sided + bootstrap| Variance comparison     |
| **H5**     | Q-learning higher long-run reward | Wilcoxon + bootstrap         | Reward optimality       |

## PLOTS ##
1. Cliff fall rate plots
Shows how often the agent falls over time.
Downward slope = learning
Lower curve = safer behavior

2. Reward plots
Reward trending upward (toward 0) reflects improved policy.

3. Exponential decay plot
Shows fitted learning curves and decay parameters b.
Higher b → faster learning.

## NOTES ##
1) The save_metrics_csv function in Main.py is designed to create csv files for metrics and append to it. If you run multiple iterations or set --num-runs X, it will continue to append to the existing csv files. Analyze.py is designed to perform a groupby function across all iterations within the csv files to take the respective average for the metrics, but will not perform any analysis if there are no generated csv files.

2) Default Modes & Hyperparameters
Baseline = what we think the minimal performing config is and provides value to our experiment
Finetuned = what we think the optimal is and provides value to our experiment

"Baseline": {

   "Q-Learning": dict(gamma=0.70, alpha=0.5, epsilon=1.0, decay_rate=0.9995, min_eps=0.10),
   
   "SARSA":      dict(gamma=0.70, alpha=0.5, epsilon=1.0, decay_rate=0.9995, min_eps=0.10),
    },

"Finetuned": {
   
   "Q-Learning": dict(gamma=0.99, alpha=0.15, epsilon=1.0, decay_rate=0.997, min_eps=0.01),
   
   "SARSA":      dict(gamma=0.99, alpha=0.12, epsilon=1.0, decay_rate=0.997,  min_eps=0.01),
    }

4) Notes on Analysis Warnings

During the statistical analysis phase, you may see the following warnings in the console. These are expected behavior given the nature of the experiment and do not indicate failures or errors.

Wilcoxon RuntimeWarning (invalid value encountered)
This occurs when Q-Learning and SARSA have identical window-level values or when all differences share the same sign (e.g., 0 when we intentionally make the baselines perform poorly). In those cases, the Wilcoxon statistic becomes numerically undefined. The analysis still completes normally, and the interpretation remains valid.

Exponential Fit Overflow Warning
The exponential decay model used in H3 may overflow when the data shows little or no decay (e.g., baseline agents that do not improve). The curve fitting step then generates overflow warnings, which are harmless and expected for flat or unstable curves.

OptimizeWarning: Covariance Could Not Be Estimated
This appears when the exponential model parameters cannot be reliably determined—typically when learning is extremely fast (finetuned agents) or nonexistent (baseline agents). The analysis continues using the best-fit values available.

These warnings do not affect report generation, statistical conclusions, or hypothesis test results. They reflect edge cases inherent to reinforcement learning performance curves.

4) How we obtained the data and analysis reports in the archived folder:
- python src\main.py --num-runs 10 --train-only
- python src\main.py --analysis-only

## DISCLAIMER ##
This project was a collaboration between four graduate-level Data Science students from UC Irvine for a class project. It was designed strictly for academic and educational purposes to explore reinforcement learning algorithms, statistical hypothesis testing, and experimental evaluation pipelines. The environments, algorithms, analyses, and results included in this repository should not be interpreted as production-grade systems, safety-critical decision tools, or authoritative benchmarks. All conclusions are limited to the assumptions, hyperparameters, and experimental conditions used in this course project.

The authors make no guarantees regarding the accuracy, completeness, or real-world reliability of the code or results. Any reuse, modification, or extension of this work is done at the user's own risk.
