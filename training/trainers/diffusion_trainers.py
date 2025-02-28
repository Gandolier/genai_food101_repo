import torch
from torchvision.utils import save_image
import os

from utils.class_registry import ClassRegistry
from utils.data_utils import del_files
from utils.diffusion_utils import DiffusionUtils, EMAHelper
from training.trainers.base_trainer import BaseTrainer

from models.ddpm_model import diffusion_models_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder



diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_init_weights(self, net):
        if isinstance(net, torch.nn.Linear):
            torch.nn.init.xavier_normal_(net.weight)
            net.bias.data.fill_(0.01)
    
    def setup_models(self):
        model_instance = diffusion_models_registry[self.config.train.model]
        self.model = model_instance(**self.config.model_args).to(self.device)
        
        if os.path.exists(self.model_checkpnt_pth):
            self.model.load_state_dict(torch.load(self.model_checkpnt_pth, 
                                                  weights_only=True), strict=False)
        else:
            self.model.apply(self.setup_init_weights)
        
        self.diffusion_utils = DiffusionUtils(**self.config.utils.diffusion)
        self.w = self.config.model_args.w
        return self.model


    def setup_optimizers(self):
        self.opt_name = self.config.train.optimizer
        self.opt_conf = self.config.optimizer_args[self.opt_name]   
        self.opt_instance = optimizers_registry[self.opt_name](self.model.parameters(), **self.opt_conf)
        self.opt = self.opt_instance.forward()

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, **self.config.optimizer_args.scheduler
            )

        if os.path.exists(self.adam_checkpnt_pth):
            self.opt.load_state_dict(torch.load(self.adam_checkpnt_pth, weights_only=True))
        return self.opt


    def setup_losses(self):
        self.loss_builder = DiffusionLossBuilder(self.config)

    def setup_ema(self):
        if self.config.ema.is_active:
            self.ema_helper = EMAHelper(mu=self.config.ema.decay)
            self.ema_helper.register(self.model)

            if os.path.exists(self.ema_checkpnt_pth):
                self.ema_helper.load_state_dict(torch.load(self.ema_checkpnt_pth))

        else:
            self.ema_helper = None

    def to_train(self):
        self.model.train()
        
    def to_eval(self):
        self.model.eval()


    def train_step(self):
        img, lbl = next(self.train_dataloader).values()
        lbl = lbl.to(self.device)
        img = img.to(self.device)
        t = self.diffusion_utils.sampleTimestep(size=img.shape[0])

        noisy_img_t, noise = self.diffusion_utils.noiseImage(img, t)
        if self.w == 0:
            pred = self.model(noisy_img_t, t, lbl)       
        else: 
            pred_uncond = self.model(noisy_img_t, t)
            pred_cond = self.model(noisy_img_t, t, lbl)
            pred = (1 + self.w) * pred_cond + (self.w) * pred_uncond   

        loss_batch = {'real_noise': noise, 'predicted_noise': pred}
        _, loss_dict = self.loss_builder.calculate_loss(loss_batch)
        step_loss = (noise - pred).square().sum(dim=(1, 2, 3)).mean(dim=0)

        self.opt.zero_grad()
        step_loss.backward()
        self.opt.step()

        if self.config.ema.is_active:
            self.ema_helper.update(self.model)

        return loss_dict


    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.model_checkpnt_pth)
        torch.save(self.opt.state_dict(), self.adam_checkpnt_pth)
        if self.config.ema.is_active:
            torch.save(self.ema_helper.state_dict(), self.ema_checkpnt_pth)


    def synthesize_images(self, batch_size, task='validation'):
        if task == 'validation':
            if self.config.ema.is_active:
                emodel = self.ema_helper.ema_copy(self.model, self.config.model_args, self.device)
        
            labels = torch.randint(0, 101, (batch_size,)).to(self.device) 
            x = torch.rand(batch_size, 3, 64, 64).to(self.device)       
            
            img_sample = self.diffusion_utils.sample(x, emodel, labels, steps='ddpm').cpu()
            img_sample = (img_sample.clamp(-1, 1) + 1) / 2
            
            path_to_saved_pics = self.experiment_dir + '/data/validation/fake'
            for i, img in enumerate(img_sample):
                save_image(img, path_to_saved_pics + '/' + str(i) + '.jpg')
            del emodel 
        
        elif task == 'inference': 
            del_files(self.infer_fake)
            batch_size = 20 # the size of the test data for each class
            num_classes = 101 # number of classes

            if self.config.ema.is_active:
                self.ema_helper.ema(self.model)
            
            labels = torch.tensor([clas for clas in self.classes.keys() for _ in range(batch_size)])
            x = torch.rand(batch_size * num_classes, 3, 64, 64).to(self.device)

            img_sample = self.diffusion_utils.sample(x, self.model, labels, steps='ddpm').cpu()
            img_sample = (img_sample.clamp(-1, 1) + 1) / 2
            for j, img in enumerate(img_sample):
                save_image(img, self.infer_fake + '/' + str(j) + '.jpg')
            
            path_to_saved_pics = self.infer_fake

        return img_sample, path_to_saved_pics, labels