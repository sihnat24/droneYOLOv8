import csv
import sys
import os
import glob

def get_latest_run(runs_dir="runs/train"):
    """Get the most recently modified run directory"""
    runs = glob.glob(os.path.join(runs_dir, "*/results.csv"))
    if not runs:
        print("No runs found")
        sys.exit(1)
    return max(runs, key=os.path.getmtime)


def compute_score(results_csv):
    """Read last epoch and compute composite ISR score"""
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows in results.csv")
        sys.exit(1)

    last = rows[-1]

    precision = float(last["metrics/precision(B)"].strip())
    recall    = float(last["metrics/recall(B)"].strip())
    map50     = float(last["metrics/mAP50(B)"].strip())

    score = 0.75 * recall + 0.25 * precision

    print(f"precision : {precision:.4f}")
    print(f"recall    : {recall:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"---")
    print(f"score     : {score:.4f}")

    return score


if __name__ == "__main__":
    # allow passing a specific results.csv or auto-detect latest run
    if len(sys.argv) > 1:
        results_csv = sys.argv[1]
    else:
        results_csv = get_latest_run()
        print(f"reading: {results_csv}\n")

    compute_score(results_csv)