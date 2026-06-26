import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def read_train_val_log(file_path: Path) -> pd.DataFrame:
    """
    Read a training log text file.

    Expected format:
        Epoch Train_ACC Train_Loss Val_ACC Val_Loss
        1 0.5532 1.1265 0.2500 1.5839
        ...

    Accuracy values are assumed to be in [0, 1] and converted to [%].
    """
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        # Skip header
        header = f.readline().strip()

        for line in f:
            parts = line.strip().split()

            # Ignore empty or incomplete lines
            if len(parts) < 5:
                continue

            try:
                epoch = int(parts[0])
                train_acc = float(parts[1]) * 100.0
                train_loss = float(parts[2])
                val_acc = float(parts[3]) * 100.0
                val_loss = float(parts[4])
            except ValueError:
                # Ignore lines such as:
                # 1000 Epoch runtimes: ...
                continue

            rows.append(
                {
                    "Epoch": epoch,
                    "Train_ACC": train_acc,
                    "Train_Loss": train_loss,
                    "Val_ACC": val_acc,
                    "Val_Loss": val_loss,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["Epoch", "Train_ACC", "Train_Loss", "Val_ACC", "Val_Loss"]
    )

    if df.empty:
        raise ValueError(
            f"No valid epoch rows were read from: {file_path}\n"
            f"First line/header was: {header}\n"
            "Please check the file format and log_dir."
        )

    return df


def plot_9_subjects_2x2(
    log_dir: str,
    output_dir: str,
    file_pattern: str = "Train_Acc_loss_log_{}.txt",
    n_subjects: int = 9,
):
    """
    Create a 2x2 figure:
        Top-left     : Training accuracy for 9 subjects
        Top-right    : Training loss for 9 subjects
        Bottom-left  : Validation accuracy for 9 subjects
        Bottom-right : Validation loss for 9 subjects

    Outputs:
        train_val_acc_loss_9_subjects_2x2.png
        train_val_acc_loss_9_subjects_2x2.pdf
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dfs = []

    for subject_id in range(1, n_subjects + 1):
        file_path = log_dir / file_pattern.format(subject_id)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = read_train_val_log(file_path)
        subject_dfs.append(df)

    # Font sizes
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 13
    legend_fontsize = 12
    suptitle_fontsize = 20

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)

    ax_train_acc = axes[0, 0]
    ax_train_loss = axes[0, 1]
    ax_val_acc = axes[1, 0]
    ax_val_loss = axes[1, 1]

    for subject_id, df in enumerate(subject_dfs, start=1):
        label = f"S{subject_id}"

        ax_train_acc.plot(
            df["Epoch"],
            df["Train_ACC"],
            linewidth=1.4,
            label=label,
        )

        ax_train_loss.plot(
            df["Epoch"],
            df["Train_Loss"],
            linewidth=1.4,
            label=label,
        )

        ax_val_acc.plot(
            df["Epoch"],
            df["Val_ACC"],
            linewidth=1.4,
            label=label,
        )

        ax_val_loss.plot(
            df["Epoch"],
            df["Val_Loss"],
            linewidth=1.4,
            label=label,
        )

    # Titles
    ax_train_acc.set_title("Training Accuracy", fontsize=title_fontsize)
    ax_train_loss.set_title("Training Loss", fontsize=title_fontsize)
    ax_val_acc.set_title("Validation Accuracy", fontsize=title_fontsize)
    ax_val_loss.set_title("Validation Loss", fontsize=title_fontsize)

    ax_train_acc.set_ylim(50, 100)
    ax_val_acc.set_ylim(50, 100)

    # Axis labels
    ax_train_acc.set_ylabel("Accuracy (%)", fontsize=label_fontsize)
    ax_val_acc.set_ylabel("Accuracy (%)", fontsize=label_fontsize)

    ax_train_loss.set_ylabel("Loss", fontsize=label_fontsize)
    ax_val_loss.set_ylabel("Loss", fontsize=label_fontsize)

    ax_val_acc.set_xlabel("Epoch", fontsize=label_fontsize)
    ax_val_loss.set_xlabel("Epoch", fontsize=label_fontsize)

    # Formatting
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.legend(
            fontsize=legend_fontsize,
            ncol=3,
            frameon=True,
        )

    # fig.suptitle(
    #     "Training and Validation Accuracy/Loss for 9 Subjects",
    #     fontsize=suptitle_fontsize,
    # )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "train_val_acc_loss_9_subjects_2x2.png"
    pdf_path = output_dir / "train_val_acc_loss_9_subjects_2x2.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    # ログファイルがあるディレクトリ
    dataset_name = "BCI2a"  # "BCI2a" or "BCI2b"
    log_dir = f"./results/Within_Subj_results_replicate/{dataset_name}_train_val_results/Model_CLT/aug_3/seed_1/"

    # 図を保存するディレクトリ
    output_dir = f"./results/learning_curve_plots/{dataset_name}/"

    plot_9_subjects_2x2(
        log_dir=log_dir,
        output_dir=output_dir,
        file_pattern="Train_Acc_loss_log_{}.txt",
        n_subjects=9,
    )