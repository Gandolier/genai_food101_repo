import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader 

from abc import abstractmethod
import random
import os
import shutil
import time

from training.loggers import TrainingLogger
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from utils.data_utils import class2idx, del_files


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device
        
        if len(os.listdir(config.train.checkpoint_path)) != 0:
            self.start_step = config.checkpoint.start_step
            self.run_id = config.checkpoint.run_id
        else:
            self.start_step = config.train.start_step
            

    @abstractmethod
    def setup_models(self):
        pass

    @abstractmethod
    def setup_optimizers(self):
        pass
    
    @abstractmethod
    def setup_losses(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass    

    def setup(self):
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_metrics()
        self.setup_ema()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()


    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()


    def setup_experiment_dir(self):
        self.experiment_dir = self.config.exp.experiment_dir

        self.infer_real = self.config.data.input_val_dir + '/' + 'testpile'
        self.infer_fake = self.config.data.input_val_dir + '/' + 'fakepile'

        self.model_checkpnt_pth = self.config.train.checkpoint_path + '/' + self.config.train.model + '.pth'
        self.adam_checkpnt_pth = self.config.train.checkpoint_path  + '/' + 'adam.pth'
        self.ema_checkpnt_pth = self.config.train.checkpoint_path + '/' + 'ema.pth'
        
    def setup_metrics(self): 
        self.metrics_instance = metrics_registry[self.config.train.val_metrics[0]]()
        return self.metrics_instance 

    def setup_logger(self):
        if os.path.exists(self.model_checkpnt_pth):
            self.logger = TrainingLogger(self.config, self.run_id)
        else:
            self.logger = TrainingLogger(self.config)
        return self.logger
        
    def setup_datasets(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.train_dataset = datasets_registry['dataset'](
            root=self.config.data.input_train_dir, 
            transforms=self.transform
            )
        self.classes = class2idx(self.config.data.input_train_dir, True)
        return self.train_dataset
    
    def setup_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.num_workers
            )
        self.train_dataloader = iter(self.train_dataloader)
        return self.train_dataloader
        

    def training_loop(self):
        self.to_train()

        for self.step in range(self.start_step, self.config.train.steps + 1):            
            losses_dict = self.train_step()
            self.logger.update_losses(losses_dict)

            if self.step % self.config.train.checkpoint_step == 0 and self.step != 0:
                self.save_checkpoint()

            if self.step % self.config.train.val_step == 0 and self.step != 0:
                val_metrics_dict, images, labels = self.validate() 

                self.logger.log_val_metrics(val_metrics_dict, step=self.step)
                self.logger.log_batch_of_images(images, 
                                                step=self.step, 
                                                images_type=[self.classes[lbl] for lbl in labels.tolist()]
                                                )
            if self.step % self.config.train.log_step == 0:
                self.logger.log_train_losses(self.step)
            
            if self.step % self.config.optimizer_args.scheduler.T_max == 0 and self.step != 0:
                self.lr_scheduler.step()


    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass


    @torch.no_grad()
    def validate(self):
        self.to_eval()            
        images_sample, images_pth, labels = self.synthesize_images(task = 'validation',
                                            batch_size = self.config.data.val_batch_size)
        real_images_pth = self.experiment_dir + '/data/validation/real'
        
        for lbl in labels.tolist():
            filename = random.choice(os.listdir(self.config.data.input_train_dir + '/' + self.classes[lbl]))
            shutil.copy(self.config.data.input_train_dir + '/' + self.classes[lbl] + '/' + filename, 
                        real_images_pth
                        )
            
        metrics_dict = {}
        metric = self.metrics_instance 
        metrics_dict[metric.get_name()] = metric(
            orig_pth=real_images_pth, 
            synt_pth=images_pth,
            fid_config=self.config.fid_args
            )
        del_files(real_images_pth)
        del_files(images_pth)
        return metrics_dict, images_sample, labels


    @abstractmethod
    def synthesize_images(self):
        pass


    @torch.no_grad()
    def inference(self): 
        self.to_eval() 
        images_sample, images_pth, labels = self.synthesize_images(batch_size = 20, 
                                                                task = 'inference') 
        metrics_dict = {}
        metric = self.metrics_instance 
        metrics_dict[metric.get_name()] = metric(
            orig_pth=self.infer_real, 
            synt_pth=images_pth,
            fid_config=self.config.fid_args
            )
        return metrics_dict, images_sample, labels 