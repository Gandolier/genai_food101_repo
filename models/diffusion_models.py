import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry


diffusion_models_registry = ClassRegistry()

@diffusion_models_registry.add_to_registry(name="initial_diffusion_model")
class VerySimpleUnetBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.class_emb_size = model_config['class_emb_size']
        self.in_channels = model_config['in_channels'] + self.class_emb_size
        self.out_channels = model_config['out_channels']
        self.conv_size = model_config['conv_size']
        self.num_classes = model_config['num_classes']
        
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(self.in_channels, self.conv_size, kernel_size=5, padding=2),
                nn.Conv2d(self.conv_size, self.conv_size * 2, kernel_size=5, padding=2),
                nn.Conv2d(self.conv_size * 2, self.conv_size * 2, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(self.conv_size * 2, self.conv_size * 2, kernel_size=5, padding=2),
                nn.Conv2d(self.conv_size * 2, self.conv_size, kernel_size=5, padding=2),
                nn.Conv2d(self.conv_size, self.out_channels, kernel_size=5, padding=2),
            ]
        )
        self.activation = nn.SiLU() 
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

        self.class_emb = nn.Embedding(self.num_classes, self.class_emb_size)


    def forward(self, x, class_labels):
        bs, ch, w, h = x.shape # size (bs, 3, 64, 64)
        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(
            bs, class_cond.shape[1], w, h
        ) # size (bs, class_emb_size, 64, 64), ready to be concat
        x = torch.cat((x, class_cond), 1) # class-conditioning

        h = []
        for i, l in enumerate(self.down_layers):
            x = self.activation(l(x)) 
            if i < len(self.down_layers) - 1: 
                h.append(x)  
                x = self.downscale(x)  

        for i, l in enumerate(self.up_layers):
            if i > 0:  
                x = self.upscale(x) 
                x += h.pop()  
            x = self.activation(l(x))  
        return x
 
    
def noising(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """
    --------------------------------------------------------------------------
    Corrupt the input `x` by mixing it with noise according to `amount`
    :param amount: torch.Tenosr of shape [x.shape[1], x.shape[2], x.shape[3]]
    --------------------------------------------------------------------------
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount