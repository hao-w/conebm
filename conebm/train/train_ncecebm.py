import os
import torch
import argparse
from sebm.sgld import *
from sebm.data import setup_data_loader
from sebm.utils import set_seed, create_exp_name, init_models, save_models, Trainer


        
class Train_NCECEBM(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, num_classes, sgld_sampler, lr, sgld_steps, img_sigma, reg_alpha):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.num_classes = num_classes
        self.sgld_sampler = sgld_sampler 
        self.sgld_steps = sgld_steps
        self.img_sigma = img_sigma
        self.reg_alpha = reg_alpha
        self.optimizer = torch.optim.Adam(list(self.models['cebm'].parameters()), lr=lr)
        self.metric_names = ['E_div', 'E_data', 'E_model']

    def train_epoch(self, epoch):
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images_fg, images_bg, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad() 
            images_bg = (images_bg + self.img_sigma * torch.randn_like(images_bg)).to(self.device)
            images_fg = (images_fg + self.img_sigma * torch.randn_like(images_fg)).to(self.device)
            labels_bg = torch.nn.functional.one_hot(torch.zeros(len(images_bg), device=self.device, dtype=torch.long), num_classes=self.num_classes)
            labels_fg = torch.nn.functional.one_hot(torch.ones(len(images_fg), device=self.device, dtype=torch.long), num_classes=self.num_classes)
            loss, metric_epoch = self.loss(torch.cat((images_fg, images_bg), 0), 
                                           torch.cat((labels_fg, labels_bg), 0), 
                                           metric_epoch)
                
            if metric_epoch['E_div'].abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
            loss.backward()
            self.optimizer.step()
        if epoch == (self.num_epochs-1):
            try:
                torch.save(self.sgld_sampler.buffer.cpu().detach(), "./weights/buffer-{:s}".format(self.exp_name))
            except:
                buffer = {'fg' : self.sgld_sampler.buffer_fg.cpu().detach(),
                          'bg' : self.sgld_sampler.buffer_bg.cpu().detach(),
                         }
                torch.save(buffer, "./weights/buffer-{:s}".format(self.exp_name))
        return {k: (v.item() / (b+1)) for k, v in metric_epoch.items()}

    
    def loss(self, data_images, labels, metric_epoch, pcd=True, init_samples=None):
        E_data = self.models['cebm'].cond_energy(data_images, labels)
        simulated_images = self.sgld_sampler.cond_sample(self.models['cebm'], labels, len(data_images), self.sgld_steps, pcd=pcd, init_samples=init_samples)
        E_model = self.models['cebm'].cond_energy(simulated_images, labels)
        E_div = E_data.mean() - E_model.mean() 
        loss = E_div + self.reg_alpha * ((E_data**2).mean() + (E_model**2).mean())
        metric_epoch['E_div'] += E_div.detach()
        metric_epoch['E_data'] += E_data.mean().detach()
        metric_epoch['E_model'] += E_model.mean().detach()
        return loss, metric_epoch

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': True}
    train_loader, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
    num_classes = 2
    
    network_args = {'device': device,
                    'num_classes': num_classes,
                    'im_height': im_h, 
                    'im_width': im_w, 
                    'input_channels': im_channels, 
                    'channels': eval(args.channels), 
                    'kernels': eval(args.kernels), 
                    'strides': eval(args.strides), 
                    'paddings': eval(args.paddings),
                    'hidden_dims': eval(args.hidden_dims),
                    'latent_dim': args.latent_dim,
                    'activation': args.activation}
    
    models = init_models(args.model_name, device, network_args)
    
    print('Start Training..')
    sgld_args = {'im_h': im_h, 
                 'im_w': im_w, 
                 'im_channels': im_channels,
                 'device': device,
                 'alpha': args.sgld_alpha,
                 'noise_std': args.sgld_noise_std,
                 'buffer_size': args.buffer_size,
                 'reuse_freq': args.reuse_freq}
    
    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'num_classes': num_classes,
                    'sgld_sampler': SGLD_Sampler2(**sgld_args),
                    'lr': args.lr,
                    'sgld_steps': args.sgld_steps,
                    'img_sigma': args.img_sigma,
                    'reg_alpha': args.reg_alpha}
    trainer = Train_NCECEBM(**trainer_args)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Train_NCECEBM', choices=['Train_NCECEBM'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--exp_id', default=None, type=str)
    ## data config
    parser.add_argument('--data', default='grassymnist_mnist', choices=['grassymnist_mnist', 'grassymnist_grass'])
    parser.add_argument('--img_sigma', default=3e-2, type=float)
#     parser.add_argument('--no_digit', default=False, action='store_true')
    ## optim config
    parser.add_argument('--lr', default=5e-5, type=float)
    ## arch config 
    parser.add_argument('--channels', default="[32,32,64,64]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--hidden_dims', default="[256]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='SiLU')
    parser.add_argument('--leak_slope', default=0.1, type=float, help='parameter for LeakyReLU activation')
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    ## sgld sampler config
    parser.add_argument('--sep_buffer', default=False, action='store_true')
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--reuse_freq', default=0.95, type=float)
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_alpha', default=2.0, type=float, help='step size is half of this value')
    parser.add_argument('--sgld_steps', default=60, type=int)
    parser.add_argument('--reg_alpha', default=5e-3, type=float)    
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)
