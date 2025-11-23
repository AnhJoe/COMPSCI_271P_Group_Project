import os
from analyze import run_analysis

BASE_BASELINE = os.path.join("data", "Baseline")
BASE_FINETUNED = os.path.join("data", "Finetuned")

layouts = [
    "CliffGauntlet",
    "DoubleCanyon",
    "OpenDesert"
]

algos = ["Q-Learning", "SARSA"]

def run_all():

    print("\nRunning ANALYSIS for BASELINE\n")
    for layout in layouts:
        q_path = os.path.join(BASE_BASELINE, f"{layout}_Q-Learning_metrics.csv")
        s_path = os.path.join(BASE_BASELINE, f"{layout}_SARSA_metrics.csv")

        out = os.path.join("analysis_output", "Baseline", layout)
        os.makedirs(out, exist_ok=True)

        print(f"\n→ Baseline: {layout}")
        run_analysis(
            q_csv_path=q_path,
            s_csv_path=s_path,
            output_dir=out,
            prefix=f"Baseline_{layout}"
        )

    print("\nRunning Analysis for Finetuned\n")
    for layout in layouts:
        q_path = os.path.join(BASE_FINETUNED, f"{layout}_Q-Learning_metrics.csv")
        s_path = os.path.join(BASE_FINETUNED, f"{layout}_SARSA_metrics.csv")

        out = os.path.join("analysis_output", "Finetuned", layout)
        os.makedirs(out, exist_ok=True)

        print(f"\n→ Finetuned: {layout}")
        run_analysis(
            q_csv_path=q_path,
            s_csv_path=s_path,
            output_dir=out,
            prefix=f"Finetuned_{layout}"
        )

    print("\nALL ANALYSIS COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    run_all()
