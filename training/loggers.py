import wandb
import torch
import os
from PIL import Image
from collections import defaultdict
import datetime

class WandbLogger:
    def __init__(self, config, id):
        #wandb.login(key=os.environ['WANDB_KEY'].strip())
        if len(os.listdir(config.train.checkpoint_path)) != 0:# -------------------- TO DO: build a checkpoint import ----------------------
            if id != None:
                self.wandb_args = {
                    "id": id,
                    "project": config.train.model,
                    "config": config
                    }
            else:
                raise ValueError("You must provide an id to resume a run from an existing checkpoint")
        else:
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.train.model,
                "name": str(datetime.datetime.now().ctime()),
                "config": config, 
                } 

        wandb.init(**self.wandb_args, resume="allow")


    @staticmethod
    def log_values(values_dict: dict, step: int):
        wandb.log(data=values_dict, step=step)
        
    @staticmethod
    def log_images(images, step: int): # images: dict of the form {'class': image}
        if not isinstance(images, dict):
            image_list = wandb.Image(data_or_path=images)   
            wandb.log({"examples": image_list}, step=step)
        else:
            image_list = [wandb.Image(data_or_path=image, caption=caption) for caption, image in images.items()]
            wandb.log({"examples": image_list}, step=step)
    

class TrainingLogger:
    def __init__(self, config, id=None): 
        self.logger = WandbLogger(config, id)
        self.losses_memory = defaultdict(list) # dict of the form {'mse': [], 'smooth_l1': []}


    def log_train_losses(self, step: int):
        avg_losses = {
            name: sum(values) / len(values) for name, values in self.losses_memory.items()
            } 
        self.logger.log_values(avg_losses, step)       
        self.losses_memory = {loss_name: [] for loss_name in self.losses_memory.keys()}

    def log_val_metrics(self, val_metrics: dict, step: int):
        self.logger.log_values(val_metrics, step)

    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type=None):
        batch = batch.cpu().detach().numpy() * 255
        if images_type == None:
            images = batch
        else:
            images = {classes: Image.fromarray(image, mode='RGB') for (classes, image) in zip(images_type, batch)}
        self.logger.log_images(images, step)
        
    def update_losses(self, losses_dict):
        # it is useful to average losses over a number of steps rather than track them at each step
        # this makes training curves smoother
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)





