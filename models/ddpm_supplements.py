import torch
from torch import nn
from utils.class_registry import ClassRegistry

ddpm_supplements = ClassRegistry()

@ddpm_supplements.add_to_registry(name="sin_embed")
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim_size, n:int=10000): # n is set as in the paper
        assert dim_size % 2 == 0, 'dim_size should be an even number'            
        super(SinusoidalEmbedding, self).__init__()
        
        self.dim_size = dim_size
        self.n = n
        
    def forward(self, x:torch.Tensor):
        N = len(x)
        output = torch.zeros(size=(N, self.dim_size)).to(x.device)
        
        for idx in range(0, N):
            for i in range(0, self.dim_size // 2):
                emb = x[idx] / (self.n ** (2*i / self.dim_size))
                output[idx, 2 * i] = torch.sin(emb)
                output[idx, (2 * i) + 1] = torch.cos(emb)
       
        return output
    
@ddpm_supplements.add_to_registry(name="self_attention")
class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels:int, n_heads:int):
        super(ImageSelfAttention, self).__init__()
        
        self.input_channels = input_channels
        self.n_heads = n_heads
        self.layernorm = nn.LayerNorm(self.input_channels)
        self.attention = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
        
    def forward(self, x:torch.Tensor):
        # x.shape: (N, C, H, W)
        _, C, H, W = x.shape
        x = x.reshape(-1, C, H*W).permute(0, 2, 1)
        normalised_x = self.layernorm(x)
        attn_val, _ = self.attention(normalised_x, normalised_x, normalised_x)
        attn_val = attn_val + x
        attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
        return attn_val