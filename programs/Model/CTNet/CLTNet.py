import datetime

# from utils import calMetrics
import math
import os

# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import time
import warnings

from pandas import ExcelWriter
from torch.backends import cudnn
from torchinfo import summary

warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True
from itertools import cycle

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt

# from utils import numberClassChannel
# from utils import load_data_evaluate
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import functional as F


class PatchEmbeddingCNN(nn.Module):
    def __init__(
        self,
        f1=16,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=40,
    ):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(
                1, f1, (1, kernel_size), (1, 1), padding="same", bias=False
            ),  # [batch, 22, 1000]
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(
                f1,
                f2,
                (number_channel, 1),
                (1, 1),
                groups=f1,
                padding="valid",
                bias=False,
            ),  #
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d(
                (1, pooling_size1)
            ),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        # hidden_size = f2   # あるいは別に小さくしたければ別パラメータで指定 modified
        # num_layers = 2
        # self.lstm1 = nn.LSTM(f2, hidden_size, num_layers, batch_first=True)
        dropout_rate_lstm = 0.5  # modified
        self.lstm1 = nn.LSTM(16, 16, 2, batch_first=True)  # (16,16,2)
        self.dropout = nn.Dropout(dropout_rate_lstm)  # modified
        self.projection = nn.Sequential(
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)           #(b,1,22,1000)
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        # print("x after cnn_module:", x.shape)  #(b, 16, 15, 8)
        input_data = x.squeeze(2)
        # print("x after squeeze(2):", input_data.shape)

        input_data = input_data.permute(0, 2, 1)
        # print("x after permute(0, 2, 1):", input_data.shape)

        x, _ = self.lstm1(input_data)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(2)
        x = self.projection(x)
        return x


########################################################################################
# The Transformer code is based on this github project and has been fine-tuned:
#    https://github.com/eeyhsong/EEG-Conformer
########################################################################################


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        # print(emb_size)
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x.squeeze(1)
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)  # (2,2,15,8)
        out = rearrange(out, "b h n d -> b n (h d)")  # (2,15,16)
        out = self.projection(out)
        return out


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(flatten_number, n_classes))

    def forward(self, x):
        out = self.fc(x)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)

        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=2,  # modifeid from CTnet
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                ),
                emb_size,
                drop_p,
            ),
            ResidualAdd(
                nn.Sequential(
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                ),
                emb_size,
                drop_p,
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)]
        )


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(
        self,
        heads=4,
        depth=6,
        emb_size=40,
        number_channel=22,
        f1=20,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        **kwargs
    ):
        super().__init__(
            PatchEmbeddingCNN(
                f1=f1,
                kernel_size=kernel_size,
                D=D,
                pooling_size1=pooling_size1,
                pooling_size2=pooling_size2,
                dropout_rate=dropout_rate,
                number_channel=number_channel,
                emb_size=emb_size,
            ),
        )


# learnable positional embedding module
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):  # x-> [batch, embedding, length]
        x = x + self.encoding[:, : x.shape[1], :].cuda()
        return self.dropout(x)


class EEGLTransformer(nn.Module):
    def __init__(
        self,
        heads=4,
        emb_size=40,
        depth=6,
        database_type="A",
        eeg1_f1=20,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        eeg1_number_channel=22,
        flatten_eeg1=600,
        **kwargs
    ):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.cnn = BranchEEGNetTransformer(
            heads,
            depth,
            emb_size,
            number_channel=self.number_channel,
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            D=eeg1_D,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
        )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(
            self.flatten_eeg1, self.number_class
        )  # FLATTEN_EEGNet + FLATTEN_cnn_module

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # modified for input shape (batch, 1, channel, time)
        # print(x.shape)

        cnn = self.cnn(x)

        # positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        # residual connect
        features = cnn + trans
        out = self.classification(self.flatten(features))
        # return features, out #<modifeid>
        return out


def numberClassChannel(database_type):
    if database_type == "A":
        number_class = 4
        number_channel = 22
    elif database_type == "B":
        number_class = 2
        number_channel = 3
    else:  # physionet
        number_class = 4
        number_channel = 64
    #    # elif database_type=='C':
    #   elif database_type=='C':
    #        number_class = 4
    #        number_channel = 128
    return number_class, number_channel
