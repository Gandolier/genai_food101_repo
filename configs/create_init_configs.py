from omegaconf import OmegaConf 
from dataclasses import dataclass, field

@dataclass
class UnetParams:
    in_channels: int = 1
    out_channels: int = 1
    conv_size: int = 32

@dataclass
class BasicUnetConfig:
    model_args: UnetParams = field(default_factory=UnetParams)

conf = OmegaConf.structured(BasicUnetConfig)
OmegaConf.save(config=conf, f='configs/basicunet_config.yaml')   

#print(OmegaConf.load('configs/basicunet_config.yaml')['model_args']['in_channels'])