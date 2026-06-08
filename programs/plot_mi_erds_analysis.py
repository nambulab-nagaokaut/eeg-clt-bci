"""
plot_mi_erds_analysis.py

ERD/ERS analysis for motor imagery EEG datasets.

This script includes:
1. Data loading for BCI Competition IV 2a and 2b
2. ERD/ERS computation using pre-cue baseline
3. Class-wise ERD/ERS time courses at C3, Cz, C4
4. Class-wise ERD/ERS topographic maps for BCI2a
5. Figure saving as both PNG and PDF

Recommended use:
    Run this script directly from VSCode.

Author:
    For EEG-CLT analysis
"""

from pathlib import Path
import warnings

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


# ============================================================
# User settings
# ============================================================

DATASET = "BCI2a"
# DATASET = "BCI2b"

ROOT_PATH = f"./data/{DATASET}_gdf/"  
SAVE_DIR = f"./results/erds_analysis/{DATASET}/"

SUBJECTS = list(range(1, 10))
# SUBJECTS = [1]  # Use this for testing one subject only.

FS = 250.0

BASELINE = (-0.5, 0.0)
ANALYSIS_WINDOW = (0.5, 4.0)

ERDS_MODE = "percent"
# ERDS_MODE = "db"

SAVE_DPI = 300


# ============================================================
# MNE settings
# ============================================================

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings(
    "ignore",
    message="Highpass cutoff frequency 100.0 is greater than lowpass cutoff frequency 0.5, setting values to 0 and Nyquist.",
)


# ============================================================
# Channel and class definitions
# ============================================================

BCI2A_CH_NAMES = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

BCI2B_CH_NAMES = ["C3", "Cz", "C4"]

BCI2A_CLASS_NAMES = {
    0: "Left hand",
    1: "Right hand",
    2: "Feet",
    3: "Tongue",
}

BCI2B_CLASS_NAMES = {
    0: "Left hand",
    1: "Right hand",
}

BANDS = {
    "mu_8_13Hz": (8.0, 13.0),
    "beta_13_30Hz": (13.0, 30.0),
}


# ============================================================
# Data loading functions
# ============================================================

def load_bci2a_data(root_path, subject, data_type, samples="1125"):
    """
    Load BCI Competition IV 2a data.

    Args:
        root_path:
            Root directory of BCI2a dataset.
            Expected structure:
                ROOT_PATH/Data/A01T.gdf
                ROOT_PATH/Data/A01E.gdf
                ROOT_PATH/True_labels/A01T.mat
                ROOT_PATH/True_labels/A01E.mat

        subject:
            Subject number, 1 to 9.

        data_type:
            "T" or "E".

        samples:
            "1125" is used for ERD/ERS because it includes -0.5 to 4.0 s.

    Returns:
        data:
            np.ndarray, shape = [n_trials, 22, n_times]

        label:
            np.ndarray, shape = [n_trials]
            Zero-based labels.
    """
    root_path = str(root_path)

    data_path = root_path + f"Data/A{subject:02d}{data_type}.gdf"
    label_path = root_path + f"True_labels/A{subject:02d}{data_type}.mat"

    raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=False)
    raw.filter(l_freq=None, h_freq=40, verbose=False)

    events, event_ids = mne.events_from_annotations(raw, verbose=False)

    if data_type == "T":
        event_id = [event_ids[key] for key in ["769", "770", "771", "772"]]

        if samples == "1125":
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=-0.5,
                tmax=4 - 1 / 250,
                baseline=None,
                verbose=False,
            )
        else:
            raise ValueError("For ERD/ERS analysis, samples='1125' is recommended.")

        data = epochs.get_data(verbose=False)[:, :22, :]
        label = scipy.io.loadmat(label_path)["classlabel"]

    elif data_type == "E":
        event_id = [event_ids[key] for key in ["783"]]

        if samples == "1125":
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=-0.5,
                tmax=4 - 1 / 250,
                baseline=None,
                verbose=False,
            )
        else:
            raise ValueError("For ERD/ERS analysis, samples='1125' is recommended.")

        data = epochs.get_data(verbose=False)[:, :22, :]
        label = scipy.io.loadmat(label_path)["classlabel"]

    else:
        raise ValueError("data_type must be 'T' or 'E'.")

    return data, (label - 1).flatten()


def load_bci2b_data(root_path, subject, data_type, samples="1125"):
    """
    Load BCI Competition IV 2b data.

    Args:
        root_path:
            Root directory of BCI2b dataset.
            Expected structure:
                ROOT_PATH/B0101T.gdf
                ROOT_PATH/B0104E.gdf
                ROOT_PATH/True_labels/B0101T.mat

        subject:
            Subject number, 1 to 9.

        data_type:
            "T" or "E".

        samples:
            "1125" is used for ERD/ERS.

    Returns:
        data:
            np.ndarray, shape = [n_trials, 3, n_times]

        label:
            np.ndarray, shape = [n_trials]
            Zero-based labels.
    """
    root_path = str(root_path)

    data_list = []
    label_list = []

    if data_type == "T":
        sessions = [1, 2, 3]
    elif data_type == "E":
        sessions = [4, 5]
    else:
        raise ValueError("data_type must be 'T' or 'E'.")

    for session in sessions:
        data_path = root_path + f"B{subject:02d}{session:02d}{data_type}.gdf"
        label_path = root_path + f"True_labels/B{subject:02d}{session:02d}{data_type}.mat"

        raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=False)
        raw.filter(l_freq=None, h_freq=40, verbose=False)

        events, event_ids = mne.events_from_annotations(raw, verbose=False)

        if data_type == "T":
            event_id = [event_ids[key] for key in ["769", "770"]]

            if samples == "1125":
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=-0.5,
                    tmax=4 - 1 / 250,
                    baseline=None,
                    verbose=False,
                )
            else:
                raise ValueError("For ERD/ERS analysis, samples='1125' is recommended.")

            data = epochs.get_data(verbose=False)[:120, :3, :]
            label = scipy.io.loadmat(label_path)["classlabel"][:120]

        else:
            event_id = [event_ids[key] for key in ["781"]]

            if samples == "1125":
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=-1.0,
                    tmax=3.5 - 1 / 250,
                    baseline=None,
                    verbose=False,
                )
            else:
                raise ValueError("For ERD/ERS analysis, samples='1125' is recommended.")

            data = epochs.get_data(verbose=False)[:120, :3, :]
            label = scipy.io.loadmat(label_path)["classlabel"][:120]

        data_list.append(data)
        label_list.append(label)

    data_all = np.concatenate(data_list, axis=0)
    label_all = np.concatenate(label_list, axis=0)

    return data_all, (label_all - 1).flatten()


def load_subject_data(root_path, dataset, subject):
    """
    Load both training and evaluation data for one subject.

    Args:
        root_path:
            Dataset root path.

        dataset:
            "BCI2a" or "BCI2b".

        subject:
            Subject number, 1 to 9.

    Returns:
        X:
            np.ndarray, shape = [n_trials, n_channels, n_times]

        y:
            np.ndarray, shape = [n_trials]
    """
    if dataset == "BCI2a":
        x_train, y_train = load_bci2a_data(root_path, subject, "T", samples="1125")
        x_test, y_test = load_bci2a_data(root_path, subject, "E", samples="1125")

    elif dataset == "BCI2b":
        x_train, y_train = load_bci2b_data(root_path, subject, "T", samples="1125")
        x_test, y_test = load_bci2b_data(root_path, subject, "E", samples="1125")

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y


# ============================================================
# ERD/ERS functions
# ============================================================

def bandpass_filter(data, fs, fmin, fmax, order=4):
    """
    Apply Butterworth bandpass filter.

    Args:
        data:
            np.ndarray, shape = [n_trials, n_channels, n_times]

    Returns:
        filtered_data:
            np.ndarray, same shape as data
    """
    nyquist = fs / 2.0
    b, a = butter(order, [fmin / nyquist, fmax / nyquist], btype="band")
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data


def compute_band_power_timecourse(data, fs, band):
    """
    Compute instantaneous band-power time course using Hilbert transform.

    Args:
        data:
            np.ndarray, shape = [n_trials, n_channels, n_times]

        fs:
            Sampling frequency.

        band:
            Tuple such as (8.0, 13.0).

    Returns:
        power:
            np.ndarray, shape = [n_trials, n_channels, n_times]
    """
    fmin, fmax = band
    filtered = bandpass_filter(data, fs, fmin, fmax)
    analytic = hilbert(filtered, axis=-1)
    power = np.abs(analytic) ** 2
    return power


def compute_erds(power, times, baseline, mode="percent"):
    """
    Compute ERD/ERS relative to baseline.

    Args:
        power:
            np.ndarray, shape = [n_trials, n_channels, n_times]

        times:
            np.ndarray, shape = [n_times]

        baseline:
            Tuple, e.g. (-0.5, 0.0)

        mode:
            "percent" or "db"

    Returns:
        erds:
            np.ndarray, shape = [n_trials, n_channels, n_times]

    Definitions:
        percent:
            ERD/ERS [%] = (P_task - P_baseline) / P_baseline * 100

        db:
            ERD/ERS [dB] = 10 * log10(P_task / P_baseline)
    """
    baseline_mask = (times >= baseline[0]) & (times < baseline[1])

    if baseline_mask.sum() == 0:
        raise ValueError(
            f"No baseline samples found. baseline={baseline}, "
            f"time range=({times[0]:.3f}, {times[-1]:.3f})"
        )

    baseline_power = power[:, :, baseline_mask].mean(axis=-1, keepdims=True)

    if mode == "percent":
        erds = (power - baseline_power) / (baseline_power + 1e-20) * 100.0
    elif mode == "db":
        erds = 10.0 * np.log10((power + 1e-20) / (baseline_power + 1e-20))
    else:
        raise ValueError("mode must be 'percent' or 'db'.")

    return erds


# ============================================================
# Plot functions
# ============================================================

def save_figure(fig, save_base, dpi=300):
    """
    Save figure as both PNG and PDF.

    Args:
        fig:
            matplotlib Figure.

        save_base:
            Path without extension.
    """
    save_base = Path(save_base)
    fig.savefig(str(save_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(save_base) + ".pdf", bbox_inches="tight")


def make_mne_info(ch_names, fs):
    """
    Create MNE Info object with standard 10-20 montage.
    """
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types="eeg",
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info


def plot_sensorimotor_erds_timecourse(
    erds,
    labels,
    times,
    ch_names,
    class_names,
    save_dir,
    band_name,
    mode,
    channels_to_plot=("C3", "Cz", "C4"),
):
    """
    Plot class-wise ERD/ERS time course at selected channels.
    """
    fig, axes = plt.subplots(
        len(channels_to_plot),
        1,
        figsize=(9, 3 * len(channels_to_plot)),
        sharex=True,
    )

    if len(channels_to_plot) == 1:
        axes = [axes]

    class_ids = sorted(np.unique(labels).tolist())

    for ax, ch_name in zip(axes, channels_to_plot):
        if ch_name not in ch_names:
            ax.set_title(f"{ch_name} is not available")
            continue

        ch_idx = ch_names.index(ch_name)

        for class_id in class_ids:
            class_data = erds[labels == class_id, ch_idx, :]
            mean_timecourse = class_data.mean(axis=0)

            ax.plot(
                times,
                mean_timecourse,
                label=class_names.get(class_id, f"Class {class_id}"),
            )

        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.axhline(0.0, linestyle="-", linewidth=0.8)

        ax.set_title(f"{band_name} ERD/ERS at {ch_name}")

        if mode == "percent":
            ax.set_ylabel("ERD/ERS [%]")
        else:
            ax.set_ylabel("ERD/ERS [dB]")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time from cue onset [s]")
    axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()

    save_base = Path(save_dir) / f"timecourse_erds_{band_name}_C3_Cz_C4_{mode}"
    save_figure(fig, save_base, dpi=SAVE_DPI)
    plt.close(fig)


def plot_classwise_erds_topomap(
    erds,
    labels,
    times,
    ch_names,
    class_names,
    fs,
    save_dir,
    band_name,
    analysis_window,
    mode,
):
    """
    Plot class-wise ERD/ERS topography.

    For BCI2a, this is useful because it has 22 EEG channels.
    """
    info = make_mne_info(ch_names, fs)

    window_mask = (times >= analysis_window[0]) & (times <= analysis_window[1])

    if window_mask.sum() == 0:
        raise ValueError(f"No samples found in analysis_window={analysis_window}")

    class_ids = sorted(np.unique(labels).tolist())

    erds_by_class = {}

    for class_id in class_ids:
        class_data = erds[labels == class_id]

        if len(class_data) == 0:
            continue

        # Mean over trials and analysis time window.
        # Output shape: [n_channels]
        erds_by_class[class_id] = class_data[:, :, window_mask].mean(axis=(0, 2))

    all_values = np.concatenate(list(erds_by_class.values()))

    # Symmetric color scale around zero.
    abs_max = np.percentile(np.abs(all_values), 95)

    if abs_max == 0:
        abs_max = 1.0

    vlim = (-abs_max, abs_max)

    n_classes = len(erds_by_class)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))

    if n_classes == 1:
        axes = [axes]

    image = None

    for ax, (class_id, values) in zip(axes, erds_by_class.items()):
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=vlim,
            contours=0,
        )

        ax.set_title(class_names.get(class_id, f"Class {class_id}"))

    cbar = fig.colorbar(image, ax=axes, shrink=0.75)

    if mode == "percent":
        cbar.set_label("ERD/ERS [%]")
    else:
        cbar.set_label("ERD/ERS [dB]")

    fig.suptitle(
        f"Class-wise ERD/ERS topography: {band_name}, "
        f"{analysis_window[0]}-{analysis_window[1]} s",
        y=1.02,
    )

    fig.tight_layout()

    save_base = (
        Path(save_dir)
        / f"topomap_erds_{band_name}_{analysis_window[0]}_{analysis_window[1]}s_{mode}"
    )

    save_figure(fig, save_base, dpi=SAVE_DPI)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if DATASET == "BCI2a":
        ch_names = BCI2A_CH_NAMES
        class_names = BCI2A_CLASS_NAMES
    elif DATASET == "BCI2b":
        ch_names = BCI2B_CH_NAMES
        class_names = BCI2B_CLASS_NAMES
    else:
        raise ValueError("DATASET must be 'BCI2a' or 'BCI2b'.")

    x_all = []
    y_all = []

    for subject in SUBJECTS:
        print(f"Loading {DATASET}, subject {subject}")

        x_subject, y_subject = load_subject_data(
            root_path=ROOT_PATH,
            dataset=DATASET,
            subject=subject,
        )

        print(f"  X: {x_subject.shape}, y: {y_subject.shape}")

        x_all.append(x_subject)
        y_all.append(y_subject)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    print("========================================")
    print(f"Dataset: {DATASET}")
    print(f"All data shape: {x_all.shape}")
    print(f"All label shape: {y_all.shape}")
    print(f"Classes: {np.unique(y_all)}")
    print("========================================")

    n_times = x_all.shape[-1]

    # For samples='1125', the epoch length is 4.5 s.
    # BCI2a: -0.5 to 4.0 s
    # BCI2b T: -0.5 to 4.0 s
    # BCI2b E: -1.0 to 3.5 s in the original loader.
    #
    # For BCI2a this time vector is exact.
    # For BCI2b, if T and E are combined, the reference event differs slightly.
    # Therefore, for publication-level ERD/ERS in BCI2b, T/E should preferably be handled separately.
    times = np.arange(n_times) / FS - 0.5

    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"Baseline: {BASELINE[0]} to {BASELINE[1]} s")
    print(f"Analysis window: {ANALYSIS_WINDOW[0]} to {ANALYSIS_WINDOW[1]} s")
    print(f"ERD/ERS mode: {ERDS_MODE}")

    for band_name, band_range in BANDS.items():
        print("----------------------------------------")
        print(f"Processing band: {band_name}, range={band_range}")

        power = compute_band_power_timecourse(
            data=x_all,
            fs=FS,
            band=band_range,
        )

        erds = compute_erds(
            power=power,
            times=times,
            baseline=BASELINE,
            mode=ERDS_MODE,
        )

        plot_sensorimotor_erds_timecourse(
            erds=erds,
            labels=y_all,
            times=times,
            ch_names=ch_names,
            class_names=class_names,
            save_dir=save_dir,
            band_name=band_name,
            mode=ERDS_MODE,
        )

        if DATASET == "BCI2a":
            plot_classwise_erds_topomap(
                erds=erds,
                labels=y_all,
                times=times,
                ch_names=ch_names,
                class_names=class_names,
                fs=FS,
                save_dir=save_dir,
                band_name=band_name,
                analysis_window=ANALYSIS_WINDOW,
                mode=ERDS_MODE,
            )
        else:
            print("BCI2b has only 3 EEG channels. Topomap is skipped.")

    print("========================================")
    print(f"Finished. Figures were saved to: {save_dir}")
    print("PNG and PDF files were generated.")
    print("========================================")


if __name__ == "__main__":
    main()