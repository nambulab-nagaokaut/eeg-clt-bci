"""
Measure_Inference_Latency.py

This script measures:
- trainable parameter count
- GPU inference latency per trial
- CPU inference latency per trial

Latency is measured with batch size = 1 using one test trial.
Data loading and preprocessing time are excluded.
"""

import argparse
import csv
import os
import platform
import random
import time
from pathlib import Path

from Load_data import get_data
from Model.CLT.CLT import CombinedModule
from Model.CTNet.CLTNet import EEGLTransformer as CLTNet
from Model.CTNet.CTNet import EEGTransformer as CTNet
from Model.Conformer import Conformer
from Model.EEGNet import EEGNET
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import torch

# Fixed experimental settings for the trained checkpoints.
FIXED_SEED = 1
FIXED_NUM_AUGMENTS = 3

project_root = Path("/workspaces/eeg-clt-bci")
os.chdir(project_root)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data_for_latency(data_path, dataset, nSub, seed_n):
    train_data, train_label, test_data, test_label = get_data(
        data_path,
        nSub,
        dataset,
        seed_n=0,
        Shuffle=False,
    )

    if dataset == "BCI2a":
        train_data, val_data, train_label, val_label = train_test_split(
            train_data,
            train_label,
            test_size=0.25,
            stratify=train_label,
            random_state=seed_n,
        )
    elif dataset == "BCI2b":
        train_data, val_data, train_label, val_label = train_test_split(
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


def get_model(model_name, config, device):
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

    return model.to(device)


def measure_single_model_latency(
    model,
    input_tensor,
    device,
    n_warmup=100,
    n_repeat=1000,
    cpu_num_threads=1,
):
    """
    Measure inference latency for one EEG trial.

    The measured latency is the wall-clock time for one forward pass.
    Data loading and preprocessing are excluded.
    """

    if device.type == "cpu":
        torch.set_num_threads(cpu_num_threads)

    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []

    with torch.inference_mode():
        for _ in range(n_repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    timings = np.asarray(timings, dtype=np.float64)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings, ddof=1)),
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
    }


def load_checkpoint_if_available(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("[WARNING] Measuring latency with randomly initialized weights.")
        return model, "not_found_random_weights"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, "loaded"


def measure_latency_all_models(
    config,
    dataset,
    data_path,
    subject_index,
    seed,
    num_augments,
    model_names,
    n_warmup,
    n_repeat,
    cpu_num_threads,
    output_dir,
):
    set_seed(seed)

    test_data, _ = load_data_for_latency(
        data_path=data_path,
        dataset=dataset,
        nSub=subject_index,
        seed_n=seed,
    )

    one_trial = torch.tensor(test_data[:1], dtype=torch.float32)

    results = []

    devices_to_measure = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_measure.insert(0, torch.device("cuda:0"))

    for model_name in model_names:
        print(f"\nMeasuring latency: {model_name}")

        model = get_model(
            model_name=model_name,
            config=config,
            device=torch.device("cpu"),
        )

        n_params = count_trainable_parameters(model)

        checkpoint_path = (
            f"./results/Within_Subj_results_replicate/"
            f"{dataset}_train_val_results/Model_{model_name}/"
            f"aug_{num_augments}/seed_{seed}/"
            f"Subject_{subject_index + 1}_best_model.pth"
        )

        model, checkpoint_status = load_checkpoint_if_available(
            model=model,
            checkpoint_path=checkpoint_path,
        )

        row = {
            "dataset": dataset,
            "subject": subject_index + 1,
            "seed": seed,
            "model": model_name,
            "checkpoint": checkpoint_status,
            "parameters": n_params,
            "input_shape": str(tuple(one_trial.shape)),
            "n_warmup": n_warmup,
            "n_repeat": n_repeat,
            "cpu_num_threads": cpu_num_threads,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA"
            ),
            "cpu_name": platform.processor(),
        }

        for device in devices_to_measure:
            stats = measure_single_model_latency(
                model=model,
                input_tensor=one_trial,
                device=device,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                cpu_num_threads=cpu_num_threads,
            )

            prefix = "gpu" if device.type == "cuda" else "cpu"

            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value

        results.append(row)
        print(row)

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        f"latency_{dataset}_subject_{subject_index + 1}_seed_{seed}.csv",
    )

    fieldnames = sorted({key for row in results for key in row.keys()})

    with open(csv_path, "w", newline="", encoding="UTF-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLatency results saved to: {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./programs/Config/BCI_2a_within.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. If not specified, config.Dataset.name is used.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Subject number, 1-based index.",
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
        help="Number of augmentations used in the checkpoint path.",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=100,
        help="Number of warm-up forward passes.",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=1000,
        help="Number of measured forward passes.",
    )
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=1,
        help="Number of CPU threads used for CPU latency measurement.",
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
        default="./results/Latency/",
        help="Directory for saving latency CSV files.",
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    dataset = args.dataset if args.dataset is not None else config.Dataset.name
    data_path = f"./data/{dataset}_gdf/"
    num_augments = args.num_augments

    subject_index = args.subject - 1

    model_names = [
        "EEGNet",
        "Conformer",
        "CTNet",
        "CLTNet",
        "CLT",
    ]

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", dataset)
    print("Subject:", args.subject)
    print("Seed:", args.seed)
    print("Num augments:", num_augments)
    print("Input config:", args.config)

    output_dir = os.path.join(args.output_dir, dataset)

    measure_latency_all_models(
        config=config,
        dataset=dataset,
        data_path=data_path,
        subject_index=subject_index,
        seed=args.seed,
        num_augments=num_augments,
        model_names=model_names,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
        cpu_num_threads=args.cpu_num_threads,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
