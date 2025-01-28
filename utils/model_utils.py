import random
import numpy as np
import torch


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
