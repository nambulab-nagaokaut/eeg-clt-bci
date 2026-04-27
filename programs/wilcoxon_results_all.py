"""
wilcoxon_results_all.py
Perform Wilcoxon signed-rank tests comparing each model against the CLT model using subject-wise seed means from the aggregated results CSV files for BCI2a, BCI2b, and Physionet datasets. The output includes mean, SD, per-subject means, and test statistics in an Excel file.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

"""
This script extends the dataset-aware Wilcoxon test to include the subject-wise
seed means as additional columns in the output table. Each model is compared
against the CLT model using the Wilcoxon signed-rank test on subject-level
averages (across seeds), and the resulting DataFrame includes the mean and
standard deviation for each model as well as the per-subject mean values.

Usage:
    Set `dataset_name` below to 'BCI2a', 'BCI2b', or 'Physionet'.
    Ensure the CSV files follow the naming pattern: summary_<dataset>_<model>_<version>_acc_results.csv.
    The version defaults to 'aug3' for BCI2a/BCI2b and 'aug1' for Physionet.
"""

# Specify the dataset: 'BCI2a', 'BCI2b', or 'Physionet'
dataset_name = "BCI2aLOSO"  # modify this as needed

# Directory containing the CSV files
file_dir = "./result_2026"

# Model identifiers to analyze
models = [
    "CLT",
    "EEGNet",
    "Conformer",
    "CTNet",
    "CLTNet",
]

# Determine version suffix based on dataset
version = "aug1" if dataset_name.lower() == "physionet" else "aug3"

# Assemble CSV filenames automatically
csv_files = [
    f"{file_dir}/summary_{dataset_name}_{model}_{version}_acc_results.csv"
    for model in models
]

# Read subject-wise data for each model
models_data = {}
subject_labels = None
for file in csv_files:
    model_name = os.path.splitext(os.path.basename(file))[0]
    if not os.path.exists(file):
        raise FileNotFoundError(f"CSV file not found: {file}")
    df = pd.read_csv(file)
    subj_cols = [col for col in df.columns if col.lower().startswith("subj")]
    fold_cols = [col for col in df.columns if col.lower().startswith("fold")]
    if subj_cols:
        values = df[subj_cols].astype(float).mean(axis=0).values * 100
        if subject_labels is None:
            subject_labels = subj_cols
    elif fold_cols:
        values = df[fold_cols].astype(float).mean(axis=0).values * 100
        if subject_labels is None:
            subject_labels = fold_cols
    else:
        if "Mean_Test_Acc" not in df.columns:
            raise ValueError(
                f"{file} does not contain subject, fold, or 'Mean_Test_Acc' columns."
            )
        values = np.array([df["Mean_Test_Acc"].astype(float).mean() * 100])
        if subject_labels is None:
            subject_labels = ["Mean_Test_Acc"]
    models_data[model_name] = values

# Identify CLT model
clt_key = None
for key in models_data:
    if "clt" in key.lower():
        clt_key = key
        break
if clt_key is None:
    raise ValueError(
        "No file name contains 'CLT'. Please include 'CLT' in the CLT model file name."
    )
clt_data = models_data[clt_key]

# Build results with subject means appended as columns
results = []
for model, data in models_data.items():
    mean_value = data.mean()
    sd_value = data.std()
    row = {
        "Model": model,
        "Mean (%)": round(mean_value, 3),
        "SD (%)": round(sd_value, 3),
    }
    # Append per-subject seed means to the row
    for label, val in zip(subject_labels, data):
        row[f"{label} (%)"] = round(val, 3)
    if model != clt_key:
        stat, p_value = wilcoxon(clt_data, data)
        row.update(
            {
                "Comparison": f"CLT vs {model}",
                "Statistic": round(stat, 3),
                "p-value": round(p_value, 5),
                "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No",
            }
        )
    else:
        row.update(
            {
                "Comparison": "CLT (self)",
                "Statistic": None,
                "p-value": None,
                "Significant (p < 0.05)": None,
            }
        )
    results.append(row)

results_df = pd.DataFrame(results)
output_path = f"./results/wilcoxon_results_subject_means_{dataset_name}.xlsx"
results_df.to_excel(output_path, index=False)

print(f"Wilcoxon test complete for {dataset_name}. Results saved to: {output_path}")
print("\n--- Results preview ---")
print(results_df)
