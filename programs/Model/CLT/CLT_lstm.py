import torch
import torch.nn as nn
from Model.CLT.Convolution import ConvModule
from Model.CLT.LSTM import LSTMLayer
from Model.CLT.Transformer import TransformerEncoder
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
import math
    
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
            nn.Linear(32, num_classes)
        )      
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  #mark
        out = self.fc(x)
        return out
    
class CombinedModule_lstm(nn.Module):
    def __init__(self, F1:int, D:int, input_size:int, hidden_size:int, EEGChans: int =22, EEGSamples: int =1000, 
                 Conv_drop:float = 0.5, numheads:int = 4, depth:int = 6, num_classes: int = 4):
        """CLT Model: Combines the Convolutional Module, LSTM Module, and Transformer Encoder Module.

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

        super(CombinedModule_lstm,self).__init__()
        self.EEGN_Conv = ConvModule(F1=F1,D=D,EEGChans=EEGChans,Conv_drop=Conv_drop)
        _,seq_length,_ = self.EEGN_Conv(torch.rand(1,EEGChans,EEGSamples)).size()
        self.LSTM_drop = nn.Dropout(p=0.15)

        self.LSTM = LSTMLayer(input_size=input_size, hidden_size=hidden_size) # normal LSTM

        self.Transformer = TransformerEncoder(embed_dim=hidden_size,numheads=numheads,depth=depth)
        self.Classify = Classifier(input_dim=hidden_size*seq_length, num_classes=num_classes)

    def forward(self,x):
        x = self.EEGN_Conv(x)
        x,_= self.LSTM(x)
        x = self.LSTM_drop(x)
        x = self.Transformer(x)
        out = self.Classify(x)
        return out


