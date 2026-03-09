"""
Check_parameters.py
Check and print the number of trainable parameters for each model (CLT, CTNet, CLTNet, EEGNet, Conformer) on BCI2a, BCI2b, and Physionet datasets.

"""

from Model.CLT.CLT import CombinedModule
from Model.CTNet.CLTNet import EEGLTransformer as CLTNet
from Model.CTNet.CTNet import EEGTransformer as CTNet
from Model.Conformer import Conformer
from Model.EEGNet import EEGNET
from omegaconf import OmegaConf
import torch
from torch import nn
from torchinfo import summary


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = "cpu"

    # BCI Competition IV 2a
    # config = OmegaConf.load("./programs/Config/BCI_2a_within.yaml")
    # n_classes, n_channels,input_samples = 4, 22, 1000 # 2a

    # config = OmegaConf.load("./programs/Config/BCI_2b_within.yaml")
    # n_classes, n_channels, input_samples = 2, 3, 1000  # 2b

    config = OmegaConf.load("./programs/Config/Physionet_LMSO.yaml")
    n_classes, n_channels, input_samples = 4, 64, 640  # physionet

    print(
        f"Number of classes: {n_classes}, channels: {n_channels}, input samples: {input_samples}"
    )

    eeg_clt = CombinedModule(**config.CLT.Model_hyperparams).to(device)
    eegnet = EEGNET(**config.EEGNet.Model_hyperparams).to(device)
    eeg_conformer = Conformer(**config.EEGConformer.Model_hyperparams).to(device)
    ctnet = CTNet(**config.CTNet.Model_hyperparams).to(device)
    cltnet = CLTNet(**config.CLTNet.Model_hyperparams).to(device)

    models = {
        "EEGNet": eegnet,
        "EEG Conformer": eeg_conformer,
        "CTNet": ctnet,
        "CLTNet": cltnet,
        "EEG-CLT": eeg_clt,
    }

    for name, m in models.items():
        n_params = count_parameters(m)
        print(f"{name}: {n_params} trainable parameters")


if __name__ == "__main__":
    main()
