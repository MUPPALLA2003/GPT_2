import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self,n_embd:int,dropout:float):

        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd,bias = False)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.fc2 = nn.Linear(4 * n_embd, n_embd,bias =False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x