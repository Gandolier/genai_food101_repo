import torch
from torch import nn
import numpy as np
from typing import Optional

class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, 
                 beta_max:float, device:str='cpu', skip_factor:float=1.,
                 beta_schedule:str='quad', skip_type:str='quad'):

        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_schedule = beta_schedule
        self.skip_type = skip_type
        self.device = device
        self.skip_factor = skip_factor

        self.betas = self.betaSamples(self.beta_schedule, self.beta_min, 
                                      self.beta_max, self.n_timesteps)
        self.betas = self.betas.float().to(self.device)

        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        
    def betaSamples(self, beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = 1 / (np.exp(-betas) + 1) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return torch.from_numpy(betas)
    
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(self.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sampleTimestep(self, size:int):
       t = torch.randint(low=0, high=self.n_timesteps, size=(size // 2 + 1,)).to(self.device)
       t = torch.cat([t, self.n_timesteps - t - 1], dim=0)[:size]
       return t
    
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        epsilon = torch.randn_like(x).to(self.device)
        alpha = (1-self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        noisy_x = x * alpha.sqrt() + epsilon * (1.0 - alpha).sqrt()
        return noisy_x, epsilon
    
    def generalized_steps(self, x, seq, model, b, 
                          label:Optional[torch.Tensor]=None):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(self.device)
                next_t = (torch.ones(n) * j).to(self.device)
                at = self.compute_alpha(b, t.long())
                at_next = self.compute_alpha(b, next_t.long())
                xt = xs[-1].to(self.device)
                if label == None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, label)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                c1 = (0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to(self.device))

        return xs[-1]
    
    def ddpm_steps(self, x, seq, model, b, 
                   label:Optional[torch.Tensor]=None):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            x0_preds = []
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(self.device)
                next_t = (torch.ones(n) * j).to(self.device)
                at = self.compute_alpha(b, t.long())
                atm1 = self.compute_alpha(b, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to(self.device)
                if label == None:
                    output = model(x, t)
                else:
                    output = model(x, t, label)
                e = output

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to(self.device))
                mean_eps = (
                    (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1, 1)
                logvar = beta_t.log()
                sample = mean + mask * torch.exp(0.5 * logvar) * noise
                xs.append(sample.to(self.device))
        return xs[-1]
    
    def sample(self, x:torch.Tensor,  
               model:torch.nn.Module,
               label:Optional[torch.Tensor]=None,
               steps:str='generalized'):
        
        if self.skip_type == "uniform":
            skip = self.n_timesteps // int(self.n_timesteps * self.skip_factor)
            seq = range(0, self.n_timesteps, skip)
        elif self.skip_type == "quad":
            seq = (
                np.linspace(0, np.sqrt(self.n_timesteps * 0.8), 
                            int(self.n_timesteps * self.skip_factor)) ** 2
                    )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        if label == None:
            if steps == 'generalized':
                x = self.generalized_steps(x, seq, model, self.betas)
            if steps == 'ddpm':
                x = self.ddpm_steps(x, seq, model, self.betas)
        else:
            if steps == 'generalized':
                x = self.generalized_steps(x, seq, model, self.betas, label)
            if steps == 'ddpm':
                x = self.ddpm_steps(x, seq, model, self.betas, label)
        return x


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module, config, device):
        module_copy = type(module)(**config).to(device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict