from torchvision.models.resnet import ResNet, BasicBlock
from models.ddpm_supplements import ddpm_supplements
from typing import Iterable, Optional
from torch import nn
from utils.class_registry import ClassRegistry
import torch

diffusion_models_registry = ClassRegistry()

class Encoder(ResNet):
    def __init__(
        self, input_channels:int, time_embedding:int, p_uncond:float,
        block=BasicBlock, block_layers:list=[2, 2, 2, 2], n_heads:int=4):
      
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.p_uncond = p_uncond
        self.input_channels = input_channels
        self.n_heads = n_heads
        self.num_classes = 101
        
        super(Encoder, self).__init__(self.block, self.block_layers)

        #class embedding layer
        self.class_embed = nn.Embedding(self.num_classes, self.time_embedding)
        
        #time embedding layer
        self.sinusiodal_embedding = ddpm_supplements['sin_embed'](self.time_embedding)
        
        fmap_channels = [64, 64, 128, 256, 512]
        #layers to project time embeddings unto feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels)
        #attention layers for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels)
        
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x:torch.Tensor, t:torch.Tensor, 
                labels:Optional[torch.Tensor]=None):
        #embed time positions
        t = self.sinusiodal_embedding(t) # (batch_size, time_embedding_size)
        if labels != None and torch.rand(1) > self.p_uncond:
            #embed labels
            labels_embedding = self.class_embed(labels) # (batch_size, time_embedding_size)       
            t = t + labels_embedding 
        #prepare fmap2
        fmap1 = self.conv1(x)
        t_emb = self.time_projection_layers[0](t)
        fmap1 = fmap1 + t_emb[:, :, None, None]
        fmap1 = self.attention_layers[0](fmap1)
        
        x = self.conv2(fmap1)
        x = self.bn1(x)
        x = self.relu(x)
        
        #prepare fmap2
        fmap2 = self.layer1(x)
        t_emb = self.time_projection_layers[1](t)
        fmap2 = fmap2 + t_emb[:, :, None, None]
        fmap2 = self.attention_layers[1](fmap2)
        
        #prepare fmap3
        fmap3 = self.layer2(fmap2)
        t_emb = self.time_projection_layers[2](t)
        fmap3 = fmap3 + t_emb[:, :, None, None]
        fmap3 = self.attention_layers[2](fmap3)
        
        #prepare fmap4
        fmap4 = self.layer3(fmap3)
        t_emb = self.time_projection_layers[3](t)
        fmap4 = fmap4 + t_emb[:, :, None, None]
        fmap4 = self.attention_layers[3](fmap4)
        
        #prepare fmap4
        fmap5 = self.layer4(fmap4)
        t_emb = self.time_projection_layers[4](t)
        fmap5 = fmap5 + t_emb[:, :, None, None]
        fmap5 = self.attention_layers[4](fmap5)
        
        return [fmap1, fmap2, fmap3, fmap4, fmap5], self.class_embed
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            ddpm_supplements['self_attention'](ch, self.n_heads) for ch in fmap_channels
        ])
        
        return layers



class DecoderBlock(nn.Module):
    def __init__(
        self, input_channels:int, output_channels:int, 
        time_embedding:int, upsample_scale:int=2, activation:nn.Module=nn.ReLU,
        compute_attn:bool=True, n_heads:int=4, p_uncond:float=.0):
        super(DecoderBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        self.p_uncond = p_uncond
        #self.num_classes = 101
        
        #attention layer
        if self.compute_attn:
            self.attention = ddpm_supplements['self_attention'](self.output_channels, self.n_heads)
        else:self.attention = nn.Identity()
        
        #class embedding layer
        #self.class_embed = nn.Embedding(self.num_classes, self.time_embedding)
        #time embedding layer
        self.sinusiodal_embedding = ddpm_supplements['sin_embed'](self.time_embedding)
        
        #time embedding projection layer
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, self.output_channels)
            )

        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale)
        
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels)

        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels)
        
        self.activation = activation()

    
    def forward(self, fmap:torch.Tensor, prev_fmap:Optional[torch.Tensor]=None, 
                class_embed:Optional[nn.Embedding]=None, t:Optional[torch.Tensor]=None, 
                labels:Optional[torch.Tensor]=None,):
        output = self.transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)
        self.class_embed = class_embed
        
        #apply residual connection with previous feature map
        if torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape'
            output = output + prev_fmap
            
        #apply timestep embedding
        if torch.is_tensor(t):
            t = self.sinusiodal_embedding(t)       
            if labels != None and torch.rand(1) > self.p_uncond:
                labels_embedding = self.class_embed(labels) # (batch_size, time_embedding_size)
                t = t + labels_embedding 
            t_emb = self.time_projection_layer(t)
            output = output + t_emb[:, :, None, None]
            
            output = self.attention(output)
            
        output = self.activation(output)
        return output


class Decoder(nn.Module):
    def __init__(
        self, last_fmap_channels:int, output_channels:int, p_uncond:float,
        time_embedding:int, first_fmap_channels:int=64, n_heads:int=4):
        super(Decoder, self).__init__()
        
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads
        self.p_uncond = p_uncond

        self.residual_layers = self.make_layers()

        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels, self.output_channels,
            time_embedding=self.time_embedding, activation=nn.Identity, 
            compute_attn=False, n_heads=self.n_heads)

        #set final layer second instance norm to identity
        self.final_layer.instance_norm2 = nn.Identity()


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None,
                labels:Optional[torch.Tensor]=None, class_embed):
        self.class_embed = class_embed
        fmaps = [fmap for fmap in reversed(fmaps)]
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                output = m(fmaps[idx], fmaps[idx+1], 
                           self.class_embed, t, labels)
                continue
            output = m(output, fmaps[idx+1], 
                       self.class_embed, t, labels)
        
        # no previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output)
        return output

      
    def make_layers(self, n:int=4):
        layers = []
        for i in range(n):
            if i == 0: in_ch = self.last_fmap_channels
            else: in_ch = layers[i-1].output_channels

            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels
            layer = DecoderBlock(
                in_ch, out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=True, n_heads=self.n_heads, 
                p_uncond=self.p_uncond)
            
            layers.append(layer)

        layers = nn.ModuleList(layers)
        return layers

  
@diffusion_models_registry.add_to_registry(name="ddpm")
class DDPM(nn.Module):
    def __init__(self, config):       
        super(DDPM, self).__init__()

        self.encoder = Encoder(**config.model_args.encoder_args)
        self.decoder = Decoder(**config.model_args.decoder_args)
    
    def forward(self, x:torch.Tensor, t:torch.Tensor, labels:torch.Tensor):
        enc_fmaps, class_embed = self.encoder(x, t=t, labels=labels)
        segmentation_mask = self.decoder(*enc_fmaps, t=t, labels=labels, 
                                         class_embed=class_embed)
        return segmentation_mask