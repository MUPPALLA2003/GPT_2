import torch
import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self,n_embd:int,vocab_length:int,seq_len:int,dropout:float):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_length,n_embd)
        self.positional_embed = nn.Embedding(seq_len,n_embd)

    def forward(self,x:torch.Tensor):

        B,T = x.shape
        positions = torch.arange(0,T,dtype = torch.long,device = x.device)

        return self.dropout(self.embedding(x) + self.positional_embed(positions).unsqueeze(0))