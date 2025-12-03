## OVERVIEW ##
This repository implements an experiment on the Gymnasium CliffWalking-v1 task environment to compare two Reinforcement Learning (RL) algorithms, Q-Learning and SARSA, across three custom CliffWalking map layouts (Cliff Gauntlet, Double Canyon, and Open Desert) and two sets of hyperparameters (Baseline and Fine-tuned). The custom environments can be found in the custom_envs.py and the hyperparameters are listed below in the notes. Our overall goal is to test our hypotheses against the expected behaviors of Q-Learning, an off-policy algorithm that prioritizes best action and converges on the optimal path, versus SARSA, an on-policy algorithm that updates on real behaviors of the agent hence converging on the safest path. 

The project includes:
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

3) How we obtained the data and analysis reports in the archived folder:
- python src\main.py --num-runs 10 --train-only
- python src\main.py --analysis-only

## DISCLAIMER ##
This project was a collaboration between four graduate-level Data Science students from UC Irvine for a class project. It was designed strictly for academic and educational purposes to explore reinforcement learning algorithms, statistical hypothesis testing, and experimental evaluation pipelines. The environments, algorithms, analyses, and results included in this repository should not be interpreted as production-grade systems, safety-critical decision tools, or authoritative benchmarks. All conclusions are limited to the assumptions, hyperparameters, and experimental conditions used in this course project.

The authors make no guarantees regarding the accuracy, completeness, or real-world reliability of the code or results. Any reuse, modification, or extension of this work is done at the user's own risk.
