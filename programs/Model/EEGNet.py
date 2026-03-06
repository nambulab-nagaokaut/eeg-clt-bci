import torch
import torch.nn as nn
from torchinfo import summary
import torch.functional as F
# Input (C,T) = (22,1000)
class EEGNET(nn.Module):
    def __init__(self, F1:int = 8, D:int = 2, drop_out = 0.5, EEGChans: int =22, EEGSamples: int =1000, num_cls:int =4):
        super(EEGNET,self).__init__()
        self.F2 = F1*D
        self.input_dim = self.F2*(EEGSamples//32)
        self.block1 = nn.Sequential(
            # F1 is filters = 8
            nn.Conv2d(in_channels=1, out_channels=F1,kernel_size=(1,64),padding="same",bias=False),
            nn.BatchNorm2d(num_features=F1),
            nn.Conv2d(in_channels= F1, out_channels=D*F1, kernel_size=(EEGChans,1), groups=F1,bias=False), #unknown depthwise_constraint in pytorch
            nn.BatchNorm2d(num_features=D*F1), #D*F1
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=drop_out) # 0.5 for within-subject or 0.25 for cross-subject
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=D*F1,out_channels=self.F2,kernel_size=(1,16),padding= "same",groups=D*F1,bias=False), #F2 = D*F1 = 2*8
            nn.Conv2d(in_channels=self.F2,out_channels=self.F2,kernel_size=(1,1),bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=drop_out)
        )
        self.fc = nn.Linear(self.input_dim,num_cls)

    def forward(self,x):
        x = x.unsqueeze(1) # Turn Input (C,T) = (22,1000) to (1,22,1000)
        # print(x.size())
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(2)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return(x)
