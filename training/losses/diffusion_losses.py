from torch import nn
from utils.class_registry import ClassRegistry


losses_registry = ClassRegistry()


@losses_registry.add_to_registry(name="mse")
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=self.reduction)

    def forward(self, batch):
        return self.loss_fn(batch["real_noise"], batch["predicted_noise"])

@losses_registry.add_to_registry(name="smooth_l1")
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=.5):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_fn = nn.SmoothL1Loss(beta=self.beta, reduction=self.reduction)

    def forward(self, batch):
        return self.loss_fn(batch["real_noise"], batch["predicted_noise"])


class DiffusionLossBuilder:
    def __init__(self, config):
        self.losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config.losses_coef.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if 'losses_args' in config and loss_name in config.losses_args:
                loss_args = config.losses_args[loss_name]           
            self.losses[loss_name] = losses_registry[loss_name](**loss_args)

    def calculate_loss(self, batch_data):
        loss_dict = {}
        total_loss = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss.forward(batch_data)
            total_loss += self.coefs[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        return float(total_loss), loss_dict