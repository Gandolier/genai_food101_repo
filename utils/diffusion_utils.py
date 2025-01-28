import torch
import numpy as np

class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, beta_max:float, device:str='cpu'):

        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        
        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
    
    
    def betaSamples(self):
        betas = []
        for i in reversed(range(self.n_timesteps)):
            T = self.n_timesteps - 1
            beta = self.beta_min + 0.5*(self.beta_max - self.beta_min) * (1 + np.cos((i/T) * np.pi))
            betas.append(beta)
            
        return torch.Tensor(betas).to(self.device)
    
    
    def sampleTimestep(self, size:int):
       return torch.randint(low=1, high=self.n_timesteps, size=(size, )).to(self.device)
    
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        # image shape: (N, C, H, W)
        # t:torch.Tensor shape: (N, )
        
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_mins_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        return (alpha_hat_sqrts * x) + (one_mins_alpha_hat_sqrt * noise), noise
    
    
    def sample(self, x:torch.Tensor, model:torch.nn.Module):
        # image shape: (N, C, H, W)
        model.eval()       
        with torch.no_grad():
            for i in reversed(range(1, self.n_timesteps)):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat
                
                predicted_noise = model(x, t)
                
                if i > 1:noise = torch.randn_like(x).to(self.device)
                else:noise = torch.zeros_like(x).to(self.device)
                
                x = 1/torch.sqrt(alpha) * (x - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
                x = x + (torch.sqrt(beta) * noise)
                
            return x
