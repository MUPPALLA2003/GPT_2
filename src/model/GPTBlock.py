import torch
import torch.nn as nn
from ResidualNet import ResidualNetwork
from Attention import CausalSelfAttention
from FeedForward import MLP

class GPTBlock(nn.Module):

        def __init__(self,n_embd:int,dropout:float,n_head:int,seq_len:int):

            super().__init__()

            self.attention = CausalSelfAttention(n_embd,n_head,seq_len,dropout)
            self.mlp = MLP(n_embd,dropout)
            self.residual_attention = ResidualNetwork(n_embd,dropout)
            self.residual_mlp = ResidualNetwork(n_embd,dropout)

        def forward(self, x:torch.Tensor):

            x = self.residual_attention(x,self.attention)
            x = self.residual_mlp(x,self.mlp)

            return x



    