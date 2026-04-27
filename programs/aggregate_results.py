"""
Aggregate_results.py

Aggregate test accuracies from multiple seeds and folds for BCI2a and BCI2b datasets, and save to CSV.
"""

import os

import pandas as pd

# ----------- Configuration ---------------
dataset_list = ["BCI2a", "BCI2b"]
dataset = dataset_list[1]
model_name_list = [
    "CLT",
    "CTNet",
    "CLTNet",
    "EEGNet",
    "Conformer",
    "CLT_lstm",
]
model_name = model_name_list[0]
type_list = ["LOSO", "Within_Subj"]
type = type_list[0]
num_augments = 3

seed_list = [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]

base_path = (
    f"./results/{type}_results_replicate/"
    f"{dataset}_train_val_results/Model_{model_name}/aug_{num_augments}/"
)
output_csv = (
    f"./results/{type}_results_replicate/"
    f"{dataset}_train_val_results/summary_"
    f"{dataset}_{model_name}_aug{num_augments}_acc_results.csv"
)

# ----------- Initialization -------------
test_records = []

try:
    # ----------- Extract Test Accuracies -------------
    for seed in seed_list:
        # Construct the path to this seed's result folder and test log
        seed_dir = os.path.join(base_path, f"seed_{seed}")
        if type == "LOSO" or type == "Physionet":
            test_log_path = os.path.join(seed_dir, "Test Results", "Test_Acc_log.txt")
        elif type == "Within_Subj":
            test_log_path = os.path.join(seed_dir, "Test_Acc_log.txt")

        if os.path.exists(test_log_path):
            with open(test_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Extract each subject's test accuracy ("Test Acc:" lines excluding "Average")
            subject_accuracies = [
                float(line.split(":")[1].strip())
                for line in lines
                if "Test Acc:" in line and "Average" not in line
            ]

            # Extract the average test accuracy line
            avg_line = next(line for line in lines if "Average Test Acc" in line)
            avg_accuracy = float(avg_line.split(":")[1].strip())

            # Record this seed's results: [seed, subj1_acc, subj2_acc, ..., mean_acc]
            test_records.append([seed] + subject_accuracies + [avg_accuracy])
        else:
            print(f"Test log not found: {test_log_path}")

    # ----------- Build DataFrame -------------
    # Assume there are 9 subjects per session
    subject_cols = [f"Subj{i+1}" for i in range(9)]
    columns = ["Seed"] + subject_cols + ["Mean_Test_Acc"]

    df = pd.DataFrame(test_records, columns=columns)

    # ----------- Compute Per-Subject Mean Across Seeds -------------
    per_subject_mean = df[subject_cols].mean(axis=0)

    # Create a summary row for means across all seeds
    mean_row = {"Seed": "MeanAcrossSeeds"}
    mean_row.update(per_subject_mean.to_dict())
    mean_row["Mean_Test_Acc"] = per_subject_mean.mean()
    print("Average Test Accuracy Across All Subjects:")
    print(per_subject_mean.mean())

    # Append the summary row
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # ----------- Save to CSV -------------
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results:  {output_csv}")
    print("----------- Done ----------\n")

except Exception as e:
    print(f"An error occurred: {e}")
