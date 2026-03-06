import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
from torch import Tensor
    
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size) #ft
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)  #it
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)   # C tilde
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size) #ot

    def forward(self, x, hidden):
        h_prev, c_prev = hidden # Hidden state and Cell state at time t-1
        combined_1 = torch.cat((x, h_prev),1)   # Input for z and o
        
        f_t = torch.sigmoid(self.forget_gate(combined_1))
        i_t = torch.sigmoid(self.input_gate(combined_1))
        c_tilde = torch.tanh(self.cell_gate(combined_1))
        o_t = torch.sigmoid(self.output_gate(combined_1))

        c_t = f_t*c_prev + i_t*c_tilde
        h_t = o_t*torch.tanh(c_t)

        return h_t,c_t

class LSTMLayer(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM_Cell = LSTMCell(input_size,hidden_size)

    def forward(self,x):
        batch_size, seq_length,_ = x.size()
        output = []
        h_t = torch.zeros(batch_size, self.hidden_size).cuda()
        c_t = torch.zeros(batch_size, self.hidden_size).cuda()

        for t in range(seq_length):
            h_t,c_t = self.LSTM_Cell(x[:,t,:], (h_t,c_t))
            output.append(h_t.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        output = output.transpose(0, 1).contiguous()
        return output,h_t