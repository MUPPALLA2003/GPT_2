import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self,config):

        super().__init__()
        self.fc1 = nn.Linear(config.n_dim, 4 * config.n_dim,bias = False)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.fc2 = nn.Linear(4 * config.n_dim, config.n_dim,bias =False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x