import pandas as pd
import os

from TRiskEvaluation import TRisk

# Configuration
datasets = ["asset", "turkcorpus"]
baselines = ["ACCESS", "Dress", "Dress-Ls", "Hybrid"]
target_method = "proposed"
criteria_list = ["meaningbert"]
alpha = 5  # Risk aversion parameter

print(f"{'Dataset':<15}\t{'Criterion':<15}\t{'Baseline':<15}\t{'Mean Baseline':<15}\t{'Mean Proposed':<15}\t{'T-Risk':<10}\t{'P-Value':<10}")

for dataset in datasets:
    for criterion in criteria_list:
        # Target file (proposed)
        target_file = f"{dataset}-{target_method}-{criterion}.csv"

        if not os.path.exists(target_file):
            # print(f"Target file not found: {target_file}")
            continue

        try:
            # Load target scores (assuming no header, single column)
            target_scores = pd.read_csv(target_file, header=None).iloc[:, 0].values
        except Exception as e:
            print(f"Error loading {target_file}: {e}")
            continue

        for baseline in baselines:
            # Baseline file
            baseline_file = f"{dataset}-{baseline}-{criterion}.csv"

            if not os.path.exists(baseline_file):
                continue

            try:
                baseline_scores = pd.read_csv(baseline_file, header=None).iloc[:, 0].values

                # Calculate T-Risk if lengths match
                if len(target_scores) == len(baseline_scores):
                    t_risk_val, p_val = TRisk(target_scores, baseline_scores, alpha)
                    print(f"{dataset:<15}\t{criterion:<15}\t{baseline:<15}\t{baseline_scores.mean():<15}\t{target_scores.mean():<15}\t{t_risk_val:<10.4f}\t{p_val:<10.4f}")
                else:
                    print(f"Size mismatch: {dataset} {baseline} {criterion}")

            except Exception as e:
                print(f"Error processing {baseline_file}: {e}")
