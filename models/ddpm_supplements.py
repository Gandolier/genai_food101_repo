import math
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
        half_dim = self.dim_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=x.device)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim_size % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


@ddpm_supplements.add_to_registry(name="self_attention")
class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels:int):
        super(ImageSelfAttention, self).__init__()
        
        self.in_channels = input_channels
        self.norm = nn.GroupNorm(num_groups=32, 
                                 num_channels=self.in_channels, 
                                 eps=1e-6, affine=True
                                 )
        self.queries = nn.Conv2d(self.in_channels, 
                                 self.in_channels, 
                                 kernel_size=1, 
                                 stride=1, 
                                 padding=0
                                 )
        self.keys = nn.Conv2d(self.in_channels, 
                              self.in_channels, 
                              kernel_size=1, 
                              stride=1, 
                              padding=0
                              )
        self.values = nn.Conv2d(self.in_channels, 
                                self.in_channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0
                                )
        self.proj_out = nn.Conv2d(self.in_channels, 
                                  self.in_channels, 
                                  kernel_size=1, 
                                  stride=1, 
                                  padding=0
                                  )
        
    def forward(self, x:torch.Tensor):
        h_ = self.norm(x)
        query = self.queries(h_)
        key = self.keys(h_)
        value = self.values(h_)

        # calculating attention
        b, c, h, w = query.shape
        query = query.reshape(b, c, h*w)
        query = query.permute(0, 2, 1)   # query shape (b, hw_q, c)
        key = key.reshape(b, c, h*w)  # key shape (b, c, hw_k)
        w_ = torch.bmm(query, key)     # w_ shape (b, hw_q, hw_k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        value = value.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # w_ shape (b, hw_k, hw_q)
        h_ = torch.bmm(value, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


@ddpm_supplements.add_to_registry(name="upsample")
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    

@ddpm_supplements.add_to_registry(name="downsample")    
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


@ddpm_supplements.add_to_registry(name="resnet_block")
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, 
                                  num_channels=self.in_channels, 
                                  eps=1e-6, affine=True
                                  )

        self.conv1 = torch.nn.Conv2d(self.in_channels,
                                     self.out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1
                                     )
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         self.out_channels
                                         )
        self.norm2 = nn.GroupNorm(num_groups=32, 
                                  num_channels=self.out_channels, 
                                  eps=1e-6, affine=True
                                  )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(self.out_channels,
                                     self.out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1
                                     )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(self.in_channels,
                                                     self.out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1
                                                     )
            else:
                self.nin_shortcut = torch.nn.Conv2d(self.in_channels,
                                                    self.out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0
                                                    )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)

        h = h + self.temb_proj(temb * torch.sigmoid(temb))[:, :, None, None]

        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h