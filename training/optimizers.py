from utils.class_registry import ClassRegistry
from torch.optim import Adam


optimizers_registry = ClassRegistry()


@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    def __init__(self, model_params, **kwargs):
        self.model_params = model_params
        self.adam_params = kwargs
        self.adam_instance = Adam
    def forward(self):
        self.adam = self.adam_instance(self.model_params, **self.adam_params)
        return self.adam        
