import torch
import torch.nn as nn
from LayerNorm import LayerNormalization

class ResidualNetwork(nn.Module):

    def __init__(self,n_embd:int,dropout:float):
        super().__init__()

        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization(n_embd)


    def forward(self,x:torch.Tensor,sublayer) -> torch.Tensor:

        return x + self.dropout(sublayer(self.layernorm(x)))   

