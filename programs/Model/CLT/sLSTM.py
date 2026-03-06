import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
import math

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size): # input_size = features
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True) #ft tilde
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)  #it tilde
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)   #zt tilde
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size,bias= True) #ot tilde

    def forward(self,x,hidden):
        # sLSTM
        h_prev, c_prev, m_prev, n_prev = hidden # Hidden state and Cell state at time t-1
        combined_1 = torch.cat((x, h_prev),1) # Input for i,f
        combined_2 = torch.cat((x, h_prev),1) # Input for z,o

        i_t = torch.exp(self.input_gate(combined_1))
        f_t = torch.exp(self.forget_gate(combined_1))
        m_t = torch.max(torch.log(f_t) + m_prev,torch.log(i_t))
        i_t = torch.exp(torch.log(i_t) - m_t)   # it'
        f_t =torch.exp(torch.log(f_t) + m_prev - m_t)   #ft'

        z_t = torch.tanh(self.cell_gate(combined_2))
        o_t = torch.sigmoid(self.output_gate(combined_2))

        c_t = f_t*c_prev + i_t*z_t
        n_t = f_t*n_prev + i_t
        h_t_tilde = c_t/n_t
        h_t = o_t*h_t_tilde

        return h_t,c_t,m_t,n_t

class sLSTMLayer(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        """LSTM Module with sLSTM core. 
        As described in https://arxiv.org/abs/2405.04517.
        Args:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
        """
        super(sLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM_Cell = sLSTMCell(input_size,hidden_size)

    def forward(self,x):
        batch_size, seq_length,_ = x.size()
        output = []
        # h_t = torch.zeros(batch_size, self.hidden_size).cuda()
        # c_t = torch.zeros(batch_size, self.hidden_size).cuda()
        # m_t = torch.zeros(batch_size, self.hidden_size).cuda()
        # n_t = torch.zeros(batch_size, self.hidden_size).cuda()
        device = x.device  # 入力テンソルに合わせたデバイス推定
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        m_t = torch.zeros(batch_size, self.hidden_size, device=device)
        n_t = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_length):
            h_t,c_t,m_t,n_t = self.LSTM_Cell(x[:,t,:],(h_t,c_t,m_t,n_t))
            output.append(h_t.unsqueeze(0))        
        output = torch.cat(output, dim=0)
        output = output.transpose(0, 1).contiguous()
        return output, h_t