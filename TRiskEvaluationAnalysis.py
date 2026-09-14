import os

import pandas as pd

from TRiskEvaluation import TRisk

# ==========================================
# Configuration
# ==========================================
DATASETS = ["asset", "turkcorpus"]
BASELINES = ["ACCESS", "Dress", "Dress-Ls", "Hybrid"]
# Composite (SARI+MB) models are the "proposed" targets, evaluated for both datasets.
TARGET_METHODS = [
    "llama1b_grpo_sari-mb",
    "llama3b_grpo_sari-mb",
    "llama1b_rloo_sari-mb",
    "llama3b_rloo_sari-mb",
]
CRITERIA = ["meaningbert"]
ALPHAS = [0, 1, 2, 3, 4, 5]  # risk aversion parameter


def find_results_dir():
    """Locate the folder holding the *-<criterion>.csv score files."""
    for d in [os.environ.get("RESULTS_DIR"), "results", "."]:
        if not d:
            continue
        for ds in DATASETS:
            for b in BASELINES:
                if os.path.exists(os.path.join(d, f"{ds}-{b}-{CRITERIA[0]}.csv")):
                    return d
    return os.environ.get("RESULTS_DIR", "results")


RESULTS_DIR = find_results_dir()
OUT_DIR = os.environ.get("TRISK_OUT_DIR", os.path.join(RESULTS_DIR, "TRisk_Output"))


def load_scores(dataset, method, criterion):
    path = os.path.join(RESULTS_DIR, f"{dataset}-{method}-{criterion}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, header=None).iloc[:, 0].values


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Results dir: {RESULTS_DIR}   ->  TRisk tables: {OUT_DIR}")

    long_rows = []  # combined long-form record of everything

    for dataset in DATASETS:
        for criterion in CRITERIA:
            for target in TARGET_METHODS:
                target_scores = load_scores(dataset, target, criterion)
                if target_scores is None:
                    print(f"[skip] target not found: {dataset}-{target}-{criterion}.csv")
                    continue

                # Table 1: t_risk, rows = baselines, cols = alpha 0..5
                trisk_df = pd.DataFrame(index=BASELINES, columns=ALPHAS, dtype=float)
                # Table 2: p_val at alpha = 0, rows = baselines
                pval0 = {}

                for baseline in BASELINES:
                    baseline_scores = load_scores(dataset, baseline, criterion)
                    if baseline_scores is None:
                        print(f"  [skip] baseline not found: {dataset}-{baseline}-{criterion}.csv")
                        continue
                    if len(baseline_scores) != len(target_scores):
                        print(f"  [warn] size mismatch: {target} ({len(target_scores)}) "
                              f"vs {baseline} ({len(baseline_scores)})")
                        continue

                    for alpha in ALPHAS:
                        t_risk_val, p_val = TRisk(target_scores, baseline_scores, alpha)
                        trisk_df.loc[baseline, alpha] = t_risk_val
                        if alpha == 0:
                            pval0[baseline] = p_val
                        long_rows.append({
                            "dataset": dataset, "criterion": criterion, "target": target,
                            "baseline": baseline, "alpha": alpha,
                            "t_risk": t_risk_val, "p_val": p_val,
                        })

                trisk_df = trisk_df.dropna(how="all")
                if trisk_df.empty:
                    continue
                trisk_df.index.name = "Baseline"
                trisk_out = trisk_df.copy()
                trisk_out.columns = [f"alpha={a}" for a in ALPHAS]

                pval_df = pd.DataFrame({"p_val (alpha=0)": pval0})
                pval_df.index.name = "Baseline"

                tag = f"{dataset}-{target}-{criterion}"
                print("\n" + "=" * 72)
                print(f"TARGET: {tag}")
                print("-- T-Risk  (rows = baseline, columns = risk aversion alpha) --")
                print(trisk_out.round(4).to_string())
                print("-- p-value at alpha=0 --")
                print(pval_df.round(6).to_string())

                trisk_out.round(6).to_csv(os.path.join(OUT_DIR, f"{tag}-trisk.csv"))
                pval_df.round(6).to_csv(os.path.join(OUT_DIR, f"{tag}-pval_alpha0.csv"))

    if long_rows:
        long_df = pd.DataFrame(long_rows)
        long_path = os.path.join(OUT_DIR, "trisk_all.csv")
        long_df.to_csv(long_path, index=False)
        print(f"\nCombined long-form results saved: {long_path}")
    else:
        print("\nNo results produced (no matching score files found).")


if __name__ == "__main__":
    main()
