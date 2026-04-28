# CLT_light.py: A lighter version of the original CLT model

import math

from Model.CLT.Convolution import ConvModule
from Model.CLT.Transformer import TransformerEncoder
from Model.CLT.sLSTM import sLSTMLayer
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Classifier(nn.Module):
    """CLTNet-like lightweight classifier.

    This classifier flattens the feature sequence and applies a single linear
    layer. It is used to evaluate whether the original EEG-CLT performance
    depends on the larger three-layer fully connected classifier.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        out = self.fc(x)
        return out


class CombinedModule(nn.Module):
    def __init__(
        self,
        F1: int,
        D: int,
        input_size: int,
        hidden_size: int,
        EEGChans: int = 22,
        EEGSamples: int = 1000,
        Conv_drop: float = 0.5,
        numheads: int = 4,
        depth: int = 6,
        num_classes: int = 4,
    ):
        """Lightweight-classifier CLT model.

        The feature extractor is the same as the original CLT model:
        CNN -> sLSTM -> Transformer.
        The classifier is changed to a CLTNet-like single linear classifier:
        Flatten -> Linear.
        Args:
            F1 (int):  F1 of Convolutional Module
            D (int): D of Convolutional Module
            input_size (int): Input size of LSTM Module
            hidden_size (int): Hidden size (Output Features Size) of LSTM Module
            EEGChans (int, optional): Number of EEG Channels. Defaults to 22.
            EEGSamples (int, optional): Number of EEG samples in a trial. Defaults to 1000.
            Conv_drop (float, optional): Dropout of Convolutional Module. Defaults to 0.5.
            numheads (int, optional): The number of heads of Multihead Attention. Defaults to 4.
            depth (int, optional): The number of Transformer Encoder Layers. Defaults to 6.
            num_classes (int, optional): Number of EEG classes (Tasks). Defaults to 4.
        """

        super(CombinedModule, self).__init__()
        self.EEGN_Conv = ConvModule(F1=F1, D=D, EEGChans=EEGChans, Conv_drop=Conv_drop)
        _, seq_length, _ = self.EEGN_Conv(torch.rand(1, EEGChans, EEGSamples)).size()
        self.LSTM_drop = nn.Dropout(p=0.15)

        self.LSTM = sLSTMLayer(input_size=input_size, hidden_size=hidden_size)  # sLSTM

        self.Transformer = TransformerEncoder(
            embed_dim=hidden_size, numheads=numheads, depth=depth
        )
        self.Classify = Classifier(
            input_dim=hidden_size * seq_length, num_classes=num_classes
        )

    def forward(self, x):
        x = self.EEGN_Conv(x)
        x, _ = self.LSTM(x)
        x = self.LSTM_drop(x)
        x = self.Transformer(x)
        out = self.Classify(x)
        return out
