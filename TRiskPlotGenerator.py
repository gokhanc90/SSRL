import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from TRiskEvaluation import TRisk

# Configuration
datasets = ["asset", "turkcorpus"]
baselines = ["ACCESS", "Dress", "Dress-Ls", "Hybrid"]
target_method = "proposed"
criteria_list = ["meaningbert"]

# Define varying alpha values from 0 to 5
alphas = np.linspace(0, 5, 50)

# Configure plot settings for high resolution
plt.rcParams.update({'font.size': 12, 'figure.dpi': 900})

print("Starting T-Risk Plot Generation...")

for dataset in datasets:
    # Create a figure with subplots (1 row, 2 columns for the criteria)
    fig, axes = plt.subplots(1, len(criteria_list), figsize=(9, 6))
    fig.suptitle(f'T-Risk Analysis for {dataset.upper()} Dataset', fontsize=16)

    # Ensure axes is iterable even if only one criterion
    if len(criteria_list) == 1:
        axes = [axes]

    for idx, criterion in enumerate(criteria_list):
        ax = axes[idx]
        target_file = f"results/{dataset}-{target_method}-{criterion}.csv"

        if not os.path.exists(target_file):
            print(f"Target file not found: {target_file}")
            continue

        try:
            # Load target scores (assuming no header, single column)
            target_scores = pd.read_csv(target_file, header=None).iloc[:, 0].values
        except Exception as e:
            print(f"Error loading {target_file}: {e}")
            continue

        for baseline in baselines:
            # Baseline file
            baseline_file = f"results/{dataset}-{baseline}-{criterion}.csv"

            if not os.path.exists(baseline_file):
                continue

            try:
                baseline_scores = pd.read_csv(baseline_file, header=None).iloc[:, 0].values

                # Calculate T-Risk if lengths match
                if len(target_scores) == len(baseline_scores):
                    trisk_values = []
                    # Compute T-Risk for each alpha value
                    for alpha in alphas:
                        t_risk_val, _ = TRisk(target_scores, baseline_scores, alpha)
                        trisk_values.append(t_risk_val)
                    
                    # Plot the line for this baseline
                    ax.plot(alphas, trisk_values, label=baseline, linewidth=1.5)
                else:
                    print(f"Size mismatch: {dataset} {baseline} {criterion}")

            except Exception as e:
                print(f"Error processing {baseline_file}: {e}")

        # Subplot formatting
        metric_name = "BERTScore" if criterion == "bertscores" else "MeaningBERT"
        ax.set_title(f'Metric: {metric_name}')
        ax.set_xlabel(r'$\alpha$ (Risk Aversion)')
        ax.set_ylabel('T-Risk Score')

        # Set y-axis to have steps of 2
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Zero line
        ax.legend(title="Baselines")
        ax.grid(True, linestyle=':', alpha=0.6)

    # Save the figure
    plt.tight_layout()
    output_filename = f"results/TRisk_Plot_{dataset}_{target_method}.eps"
    plt.savefig(output_filename, format='eps')
    print(f"Saved plot: {output_filename}")
    plt.close()

print("Processing complete.")
