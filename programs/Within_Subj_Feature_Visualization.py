"""
Within_Subj_tSNE_Feature_Visualization.py

t-SNE visualization of latent feature embeddings for within-subject MI-EEG models.

- Dataset: BCI Competition IV 2a
- Protocol: within-subject
- Subject: selectable by command-line argument
- Models: EEGNet, CLTNet, CLT (proposed EEG-CLT)
- Feature: input to the last Linear layer, i.e., embedding immediately before final classification
"""

import argparse
import os
from pathlib import Path
import random

from Load_data import get_data
from Model.CLT.CLT import CombinedModule
from Model.CTNet.CLTNet import EEGLTransformer as CLTNet
from Model.CTNet.CTNet import EEGTransformer as CTNet
from Model.Conformer import Conformer
from Model.EEGNet import EEGNET
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Fixed experimental settings.
# Fixed experimental settings for the trained checkpoints.
FIXED_SUBJECT = 1
FIXED_SEED = 1400
FIXED_NUM_AUGMENTS = 3

project_root = Path("/workspaces/eeg-clt-bci")
os.chdir(project_root)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_within_subject_test_data(data_path, dataset, n_sub_zero_based, seed_n):
    """
    Load and normalize test data using the same train-set statistics as training.

    Parameters
    ----------
    data_path : str
        Dataset path.
    dataset : str
        Dataset name, e.g., "BCI2a" or "BCI2b".
    n_sub_zero_based : int
        Subject index, zero-based.
    seed_n : int
        Random seed used for train/validation split.

    Returns
    -------
    test_data : np.ndarray
        Normalized test data.
    test_label : np.ndarray
        Test labels.
    """

    train_data, train_label, test_data, test_label = get_data(
        data_path,
        n_sub_zero_based,
        dataset,
        seed_n=0,
        Shuffle=False,
    )

    if dataset == "BCI2a":
        train_data, _, train_label, _ = train_test_split(
            train_data,
            train_label,
            test_size=0.25,
            stratify=train_label,
            random_state=seed_n,
        )
    elif dataset == "BCI2b":
        train_data, _, train_label, _ = train_test_split(
            train_data,
            train_label,
            test_size=0.2,
            stratify=train_label,
            random_state=seed_n,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_mean = np.mean(train_data, axis=(0, 2), keepdims=True)
    train_std = np.std(train_data, axis=(0, 2), keepdims=True)

    test_data = (test_data - train_mean) / train_std

    return test_data, test_label


def get_model(model_name, config):
    """
    Build model from config.
    """

    if model_name == "CLT":
        model = CombinedModule(**config.CLT.Model_hyperparams)
    elif model_name == "EEGNet":
        model = EEGNET(**config.EEGNet.Model_hyperparams)
    elif model_name == "Conformer":
        model = Conformer(**config.EEGConformer.Model_hyperparams)
    elif model_name == "CTNet":
        model = CTNet(**config.CTNet.Model_hyperparams)
    elif model_name == "CLTNet":
        model = CLTNet(**config.CLTNet.Model_hyperparams)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained checkpoint.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n"
            "Please check seed, num_augments, model name, and subject number."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def find_last_linear_layer(model):
    """
    Find the last nn.Linear layer in a model.

    We use the input to this layer as the latent embedding immediately before
    the final classification layer.
    """

    last_linear_name = None
    last_linear_module = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear_module = module

    if last_linear_module is None:
        raise RuntimeError("No nn.Linear layer was found in the model.")

    return last_linear_name, last_linear_module


def extract_embedding_before_last_linear(model, data_tensor, device, batch_size=128):
    """
    Extract latent embeddings using a forward pre-hook on the last Linear layer.

    The forward pre-hook captures the input to the final Linear layer.
    This is treated as the embedding immediately before classification.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    data_tensor : torch.Tensor
        Test data tensor.
    device : torch.device
        CPU or CUDA device.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    embeddings : np.ndarray
        Extracted embeddings, shape = (n_trials, n_features).
    logits : np.ndarray
        Model output logits, shape = (n_trials, n_classes).
    target_layer_name : str
        Name of the hooked final Linear layer.
    """

    model = model.to(device)
    model.eval()
    data_tensor = data_tensor.to(device)

    target_layer_name, target_layer = find_last_linear_layer(model)

    captured_features = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if isinstance(x, tuple):
            x = x[0]
        x = x.detach()
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        captured_features.append(x.cpu())

    handle = target_layer.register_forward_pre_hook(pre_hook)

    logits_list = []

    with torch.inference_mode():
        for start in range(0, data_tensor.shape[0], batch_size):
            batch = data_tensor[start : start + batch_size]
            outputs = model(batch)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            logits_list.append(outputs.detach().cpu())

    handle.remove()

    embeddings = torch.cat(captured_features, dim=0).numpy()
    logits = torch.cat(logits_list, dim=0).numpy()

    return embeddings, logits, target_layer_name


def run_tsne(embeddings, perplexity=30, random_state=0):
    """
    Standardize embeddings and project them to 2D using t-SNE.
    """

    embeddings = StandardScaler().fit_transform(embeddings)

    n_samples = embeddings.shape[0]
    if perplexity >= n_samples:
        adjusted_perplexity = max(5, (n_samples - 1) // 3)
        print(
            f"[WARNING] perplexity={perplexity} is too large for n_samples={n_samples}. "
            f"Using perplexity={adjusted_perplexity}."
        )
        perplexity = adjusted_perplexity

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )

    return tsne.fit_transform(embeddings)


def get_class_names(dataset):
    if dataset == "BCI2a":
        return ["Left hand", "Right hand", "Foot", "Tongue"]
    if dataset == "BCI2b":
        return ["Left hand", "Right hand"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def plot_tsne_single_model(
    tsne_xy,
    labels,
    dataset,
    model_name,
    subject_number,
    output_path_base,
    accuracy=None,
):
    """
    Plot t-SNE result for a single model.
    """

    class_names = get_class_names(dataset)

    plt.figure(figsize=(6.0, 5.2))

    for class_id, class_name in enumerate(class_names):
        idx = labels == class_id
        plt.scatter(
            tsne_xy[idx, 0],
            tsne_xy[idx, 1],
            s=28,
            alpha=0.80,
            label=class_name,
            edgecolors="none",
        )

    title = f"{model_name}, Subject {subject_number}"
    if accuracy is not None:
        title += f", Acc.={accuracy * 100:.1f}%"

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path_base + ".png", dpi=300)
    plt.savefig(output_path_base + ".pdf", bbox_inches="tight")
    plt.close()


def plot_tsne_multiple_models(
    tsne_results,
    labels,
    dataset,
    subject_number,
    output_path_base,
):
    """
    Plot t-SNE results for multiple models in one horizontal figure.

    tsne_results should be a list of dictionaries:
    [
        {"model": "EEGNet", "xy": ..., "accuracy": ...},
        ...
    ]
    """

    class_names = get_class_names(dataset)
    n_models = len(tsne_results)

    all_xy = np.vstack([item["xy"] for item in tsne_results])
    x_min, x_max = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
    y_min, y_max = np.min(all_xy[:, 1]), np.max(all_xy[:, 1])

    x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    common_xlim = (x_min - x_margin, x_max + x_margin)
    common_ylim = (y_min - y_margin, y_max + y_margin)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(5.0 * n_models, 4.6),
        squeeze=False,
    )
    axes = axes[0]

    for ax, item in zip(axes, tsne_results):
        xy = item["xy"]
        model_name = item["model"]
        accuracy = item.get("accuracy", None)

        for class_id, class_name in enumerate(class_names):
            idx = labels == class_id
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=24,
                alpha=0.80,
                label=class_name,
                edgecolors="none",
            )

        title = f"{model_name}"
        if accuracy is not None:
            title += f"\nAcc.={accuracy * 100:.1f}%"

        ax.set_title(title)
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("t-SNE dim. 1")
        ax.set_ylabel("t-SNE dim. 2")

    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=len(class_names),
        frameon=True,
    )

    fig.suptitle(f"t-SNE of latent embeddings, Subject {subject_number}", y=1.02)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def compute_row_normalized_confusion_matrix(labels, preds, dataset):
    """
    Compute raw and row-normalized confusion matrices.

    Rows indicate true classes and columns indicate predicted classes.
    The row-normalized matrix is expressed as percentages.
    """
    class_names = get_class_names(dataset)
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = (
        np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        )
        * 100.0
    )

    return cm, cm_percent


def plot_confusion_matrix_single_model(
    labels,
    preds,
    dataset,
    model_name,
    subject_number,
    output_path_base,
    normalize=True,
):
    """
    Plot a confusion matrix for a single model.

    Rows indicate true classes and columns indicate predicted classes.
    If normalize=True, values are row-normalized percentages.
    """
    class_names = get_class_names(dataset)
    cm, cm_percent = compute_row_normalized_confusion_matrix(labels, preds, dataset)
    matrix_to_plot = cm_percent if normalize else cm.astype(np.float64)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(matrix_to_plot, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)" if normalize else "Number of trials")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    accuracy = np.mean(preds == labels)
    ax.set_title(f"{model_name}, Subject {subject_number}, Acc.={accuracy * 100:.1f}%")

    threshold = matrix_to_plot.max() / 2.0 if matrix_to_plot.size > 0 else 0.0

    for i in range(matrix_to_plot.shape[0]):
        for j in range(matrix_to_plot.shape[1]):
            if normalize:
                text_value = f"{matrix_to_plot[i, j]:.1f}"
            else:
                text_value = f"{matrix_to_plot[i, j]:.0f}"

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if matrix_to_plot[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    return cm, cm_percent


def plot_confusion_matrix_multiple_models(
    confusion_results,
    dataset,
    subject_number,
    output_path_base,
    normalize=True,
):
    """
    Plot confusion matrices for multiple models in one horizontal figure.

    confusion_results should be a list of dictionaries:
    [
        {"model": "EEGNet", "labels": labels, "preds": preds, "accuracy": acc},
        ...
    ]
    """
    class_names = get_class_names(dataset)
    n_models = len(confusion_results)

    matrices = []
    for item in confusion_results:
        cm, cm_percent = compute_row_normalized_confusion_matrix(
            item["labels"],
            item["preds"],
            dataset,
        )
        matrices.append(cm_percent if normalize else cm.astype(np.float64))

    vmax = max(np.max(matrix) for matrix in matrices) if matrices else 100.0

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(4.8 * n_models, 4.4),
        squeeze=False,
    )
    axes = axes[0]

    im = None
    for ax, item, matrix_to_plot in zip(axes, confusion_results, matrices):
        im = ax.imshow(matrix_to_plot, interpolation="nearest", vmin=0, vmax=vmax)

        model_name = item["model"]
        accuracy = item.get("accuracy", None)
        title = model_name
        if accuracy is not None:
            title += f"\nAcc.={accuracy * 100:.1f}%"

        ax.set_title(title)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        threshold = vmax / 2.0 if vmax > 0 else 0.0

        for i in range(matrix_to_plot.shape[0]):
            for j in range(matrix_to_plot.shape[1]):
                if normalize:
                    text_value = f"{matrix_to_plot[i, j]:.1f}"
                else:
                    text_value = f"{matrix_to_plot[i, j]:.0f}"

                ax.text(
                    j,
                    i,
                    text_value,
                    ha="center",
                    va="center",
                    color="white" if matrix_to_plot[i, j] > threshold else "black",
                    fontsize=8,
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("Percentage (%)" if normalize else "Number of trials")

    fig.suptitle(f"Confusion matrices, Subject {subject_number}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(labels, preds, dataset, output_path_base):
    """
    Save raw and row-normalized confusion matrices as CSV files.
    """
    class_names = get_class_names(dataset)
    cm, cm_percent = compute_row_normalized_confusion_matrix(labels, preds, dataset)

    raw_path = output_path_base + "_raw.csv"
    percent_path = output_path_base + "_percent.csv"

    np.savetxt(
        raw_path,
        cm,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )
    np.savetxt(
        percent_path,
        cm_percent,
        delimiter=",",
        fmt="%.6f",
        header=",".join(class_names),
        comments="",
    )


def save_embeddings_csv(embeddings, tsne_xy, labels, preds, output_path):
    """
    Save embedding and t-SNE coordinates for reproducibility.
    """

    header = ["label", "prediction", "tsne_1", "tsne_2"]
    embedding_headers = [f"feature_{i}" for i in range(embeddings.shape[1])]
    header.extend(embedding_headers)

    data = np.column_stack([labels, preds, tsne_xy, embeddings])

    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./programs/Config/BCI_2a_within.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. If omitted, config.Dataset.name is used.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=FIXED_SUBJECT,
        help="Subject number, 1-based index. Example: 1, 2, ..., 9.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Seed corresponding to the trained checkpoint.",
    )
    parser.add_argument(
        "--num_augments",
        type=int,
        default=FIXED_NUM_AUGMENTS,
        help="Number of augmentations used in checkpoint path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EEGNet", "CLTNet", "CLT"],
        help="Models to visualize. Example: --models EEGNet CLTNet CLT",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--tsne_seed",
        type=int,
        default=0,
        help="Random seed for t-SNE.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES setting.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/tSNE/",
        help="Output directory.",
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    dataset = args.dataset if args.dataset is not None else config.Dataset.name
    data_path = f"./data/{dataset}_gdf/"
    subject_index = args.subject - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", dataset)
    print("Subject:", args.subject)
    print("Seed:", args.seed)
    print("Num augments:", args.num_augments)
    print("Models:", args.models)

    test_data, test_label = load_within_subject_test_data(
        data_path=data_path,
        dataset=dataset,
        n_sub_zero_based=subject_index,
        seed_n=args.seed,
    )

    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    labels = np.asarray(test_label)

    output_dir = os.path.join(
        args.output_dir,
        dataset,
        f"Subject_{args.subject}",
        f"seed_{args.seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    tsne_results = []
    confusion_results = []

    for model_name in args.models:
        print(f"\nProcessing model: {model_name}")

        checkpoint_path = (
            f"./results/Within_Subj_results_replicate/"
            f"{dataset}_train_val_results/Model_{model_name}/"
            f"aug_{args.num_augments}/seed_{args.seed}/"
            f"Subject_{args.subject}_best_model.pth"
        )

        print("Checkpoint:", checkpoint_path)

        model = get_model(model_name, config)
        model = load_checkpoint(model, checkpoint_path, device)

        embeddings, logits, target_layer_name = extract_embedding_before_last_linear(
            model=model,
            data_tensor=test_tensor,
            device=device,
            batch_size=args.batch_size,
        )

        preds = np.argmax(logits, axis=1)
        accuracy = np.mean(preds == labels)

        print("Target layer:", target_layer_name)
        print("Embedding shape:", embeddings.shape)
        print(f"Test accuracy: {accuracy * 100:.2f}%")

        confusion_output_base = os.path.join(
            output_dir,
            f"confusion_{dataset}_Subject_{args.subject}_{model_name}",
        )

        plot_confusion_matrix_single_model(
            labels=labels,
            preds=preds,
            dataset=dataset,
            model_name=model_name,
            subject_number=args.subject,
            output_path_base=confusion_output_base,
            normalize=True,
        )

        save_confusion_matrix_csv(
            labels=labels,
            preds=preds,
            dataset=dataset,
            output_path_base=confusion_output_base,
        )

        confusion_results.append(
            {
                "model": model_name,
                "labels": labels,
                "preds": preds,
                "accuracy": accuracy,
            }
        )

        tsne_xy = run_tsne(
            embeddings=embeddings,
            perplexity=args.perplexity,
            random_state=args.tsne_seed,
        )

        single_output_base = os.path.join(
            output_dir,
            f"tSNE_{dataset}_Subject_{args.subject}_{model_name}",
        )

        plot_tsne_single_model(
            tsne_xy=tsne_xy,
            labels=labels,
            dataset=dataset,
            model_name=model_name,
            subject_number=args.subject,
            output_path_base=single_output_base,
            accuracy=accuracy,
        )

        csv_path = os.path.join(
            output_dir,
            f"embedding_tSNE_{dataset}_Subject_{args.subject}_{model_name}.csv",
        )

        save_embeddings_csv(
            embeddings=embeddings,
            tsne_xy=tsne_xy,
            labels=labels,
            preds=preds,
            output_path=csv_path,
        )

        tsne_results.append(
            {
                "model": model_name,
                "xy": tsne_xy,
                "accuracy": accuracy,
            }
        )

    if len(tsne_results) >= 2:
        combined_output_base = os.path.join(
            output_dir,
            f"tSNE_{dataset}_Subject_{args.subject}_combined",
        )

        plot_tsne_multiple_models(
            tsne_results=tsne_results,
            labels=labels,
            dataset=dataset,
            subject_number=args.subject,
            output_path_base=combined_output_base,
        )

        print("Combined figure saved to:", combined_output_base + ".png")
        print("Combined figure saved to:", combined_output_base + ".pdf")

    if len(confusion_results) >= 2:
        combined_confusion_output_base = os.path.join(
            output_dir,
            f"confusion_{dataset}_Subject_{args.subject}_combined",
        )

        plot_confusion_matrix_multiple_models(
            confusion_results=confusion_results,
            dataset=dataset,
            subject_number=args.subject,
            output_path_base=combined_confusion_output_base,
            normalize=True,
        )

        print(
            "Combined confusion matrix saved to:",
            combined_confusion_output_base + ".png",
        )
        print(
            "Combined confusion matrix saved to:",
            combined_confusion_output_base + ".pdf",
        )

    print("\nDone.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
