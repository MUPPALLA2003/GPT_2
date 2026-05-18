import torch
import torch.nn as nn
from Embeddings import Embeddings
from GPTBlock import GPTBlock
from LayerNorm import LayerNormalization
from typing import Optional
import torch.nn.functional as F
class GPT2(nn.Module):

    def __init__(self,n_embd,seq_length,dropout,n_head,n_layer,vocab_size):

        super().__init__()

        self.embeddings = Embeddings(n_embd,vocab_size,seq_length,dropout)
        self.block = nn.ModuleList([GPTBlock(n_embd,dropout,n_head,seq_length) for _ in range(n_layer)])
        self.layernorm = LayerNormalization(n_embd)
        self.projection_layer = nn.Linear(n_embd,vocab_size,bias = False)
        self._init_weights()
        self.projection_layer.weight = self.embeddings.embedding.weight


    def _init_weights(self):
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)    

    def forward(self,x:torch.Tensor,targets:Optional[torch.Tensor] = None):

        x = self.embeddings(x)
        
        for block in self.block:
            x = block(x)

        x = self.layernorm(x)
        logits = self.projection_layer(x)
        loss = None

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

  



        