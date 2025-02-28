import torch
from torch import nn

from models.ddpm_supplements import ddpm_supplements
from utils.class_registry import ClassRegistry

diffusion_models_registry = ClassRegistry()


ResnetBlock = ddpm_supplements['resnet_block']
AttnBlock = ddpm_supplements['self_attention']
Downsample = ddpm_supplements['downsample']
Upsample = ddpm_supplements['upsample']
get_timestep_embedding = ddpm_supplements['sin_embed']

@diffusion_models_registry.add_to_registry(name="ddim")
class Model(nn.Module):
    def __init__(self, ch:int, out_ch:int, ch_mult:list, num_res_blocks:int, w, 
                 attn_resolutions:list, dropout:float, in_channels:int, p: float,
                 resolution:int, resamp_with_conv:bool, emb_dim:int, num_class:int=101):
        super().__init__()
        ch_mult = tuple(ch_mult)
        
        self.ch = ch
        self.p = p
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.get_timestep_embedding = get_timestep_embedding(self.ch)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1
                                       )

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout
                                         ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout
                                       )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout
                                       )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout
                                         ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, 
                                     num_channels=block_in, 
                                     eps=1e-6, affine=True
                                     )
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1
                                        )

        # change in embedding dimensionality
        self.emb_change = nn.Sequential(
                                        nn.SiLU(),
                                        nn.Linear(self.emb_dim, self.temb_ch),
                                        )

    def forward(self, x, t, labels=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = self.get_timestep_embedding(t)
        temb = self.temb.dense[0](temb)
        temb = temb * torch.sigmoid(temb)
        temb = self.temb.dense[1](temb)

        # class embedding and probability mask
        if labels is not None:
            class_emb = nn.Embedding(self.num_class, self.emb_dim)
            class_emb = self.emb_change(class_emb(labels))
            p_mask = torch.rand(x.shape[0]) > self.p

            # propagating temb through the model along with the temb
            temb[p_mask] += class_emb[p_mask]

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = h * torch.sigmoid(h)
        h = self.conv_out(h)
        return h