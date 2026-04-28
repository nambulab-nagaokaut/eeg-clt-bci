from Model.CLT.Convolution import ConvModule
from Model.CLT.Transformer import TransformerEncoder
from Model.CLT.sLSTM import sLSTMLayer
import torch
import torch.nn as nn


class Classifier(nn.Sequential):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class CombinedParallelModule(nn.Module):
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
        """
        CLT-parallel model with concat fusion.

        Args:
            F1 (int): Number of temporal filters in the convolution module.
            D (int): Depth multiplier in the convolution module.
            input_size (int): Feature dimension of the convolution module output.
            hidden_size (int): Hidden feature dimension used by sLSTM and Transformer.
            EEGChans (int): Number of EEG channels.
            EEGSamples (int): Number of EEG samples in a trial.
            Conv_drop (float): Dropout rate in the convolution module.
            numheads (int): Number of attention heads.
            depth (int): Number of Transformer encoder layers.
            num_classes (int): Number of MI classes.
        """

        super().__init__()

        self.EEGN_Conv = ConvModule(
            F1=F1,
            D=D,
            EEGChans=EEGChans,
            Conv_drop=Conv_drop,
        )

        with torch.no_grad():
            dummy = torch.rand(1, EEGChans, EEGSamples)
            _, seq_length, conv_feature_dim = self.EEGN_Conv(dummy).size()

        # Project CNN features to hidden_size so that the sLSTM branch and
        # Transformer branch operate on the same feature dimension.
        self.Input_proj = nn.Linear(conv_feature_dim, hidden_size)

        self.LSTM = sLSTMLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.LSTM_drop = nn.Dropout(p=0.15)

        self.Transformer = TransformerEncoder(
            embed_dim=hidden_size,
            numheads=numheads,
            depth=depth,
        )

        # Concat fusion doubles the feature dimension.
        self.Classify = Classifier(
            input_dim=2 * hidden_size * seq_length,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.EEGN_Conv(x)

        # Shared projected representation.
        x = self.Input_proj(x)

        # Parallel branches.
        x_lstm, _ = self.LSTM(x)
        x_lstm = self.LSTM_drop(x_lstm)

        x_trans = self.Transformer(x)

        # Concat fusion along the feature dimension.
        x_cat = torch.cat([x_lstm, x_trans], dim=-1)

        out = self.Classify(x_cat)
        return out


CombinedModule = CombinedParallelModule
