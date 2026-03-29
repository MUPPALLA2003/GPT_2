import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self,n_embd,n_head,seq_len,dropout):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.attn_matrix = nn.Linear(n_embd,3*n_embd)
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        self.seq_len = seq_len

        if not self.flash:
            print('flash attention requires PyTorch >= 2.0')
            self.register_buffer('mask',torch.tril(torch.ones(seq_len,seq_len)).view(1,1,seq_len,seq_len))
    
    def attention(self,q,k,v,T):

        if self.flash:
            y = F.scaled_dot_product_attention(q,k,v,attn_mask = None,dropout_p = self.dropout.p if self.training else 0,is_causal = True)

        else:
            attention_logits = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            attention_logits = attention_logits.masked_fill(self.mask[:,:,:T,:T] == 0,float('-inf'))
            attention_scores = F.softmax(attention_logits,dim = -1)
            attention_scores = self.dropout(attention_scores)
            y = attention_scores @ v
            
        return y


    def forward(self,x):

        B,T,C = x.size()
        query,key,value = self.attn_matrix(x).split(self.n_embd,dim = 2)
        q = query.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        k = key.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        v = value.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        y = self.attention(q,k,v,T)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.dropout(y)

        return y
