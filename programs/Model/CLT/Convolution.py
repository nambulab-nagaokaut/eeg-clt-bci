import torch
import torch.nn as nn

class ConvModule(nn.Module):    # 1000 samples
    def __init__(self,F1:int,D:int,EEGChans:int = 22,Conv_drop:float = 0.5):
        """Convolutional Module based on EEGNET.
        As described in https://arxiv.org/abs/1611.08024.
        Args:
            F1 (int): Output Features of Temporal Conv2D 
            D (int): Number of spatial filters in Depthwise Conv2D
            EEGChans (int): Number of EEG Channels (22 for BCI 2a, 3 for BCI 2b, 64 for Physionet). Default to 22
            Conv_drop (float): Dropout of Convolutional Module. Default to 0.5
        """
        super(ConvModule,self).__init__()
        
        "F2: Output Features of Separable Conv2D. "
        F2 = F1*D
        self.block1 = nn.Sequential(
            # F1 is filters = 8
            nn.Conv2d(in_channels=1, out_channels=F1,kernel_size=(1,125),padding="same",bias=False), #kernel size (1,125) - half sampling rate (250)
            nn.BatchNorm2d(num_features=F1),
            nn.Conv2d(in_channels= F1, out_channels=D*F1, kernel_size=(EEGChans,1), groups=F1,bias=False), #unknown depthwise_constraint in pytorch
            nn.BatchNorm2d(num_features=D*F1), #D*F1
            nn.ELU(),#
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=Conv_drop)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=D*F1,out_channels=F2,kernel_size=(1,16),padding= "same",bias=False), # 1000ms MI Activities
            # nn.Conv2d(in_channels=F2,out_channels=F2,kernel_size=(1,1),bias=False),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=Conv_drop)
        )

    def forward(self,x):
        x = x.unsqueeze(1) # Turn Input (C,T) = (22,1000) to (1,22,1000)
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(2)
        x = x.transpose(2,1) # Transpose shape from (features, seq) into (Sequence, features)
        return(x)