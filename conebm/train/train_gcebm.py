import os
import torch
import argparse
from sebm.sgld import SGLD_Sampler, SGLD_Sampler2
from sebm.data import setup_data_loader
from sebm.utils import set_seed, create_exp_name, init_models, save_models, Trainer

class Train_GCEBM(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, sgld_sampler_fg, sgld_sampler_bg, lr, sgld_steps, img_sigma, reg_alpha, full_grad, disc_gamma):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.sgld_sampler_fg = sgld_sampler_fg
        self.sgld_sampler_bg = sgld_sampler_bg
        self.sgld_steps = sgld_steps
        self.img_sigma = img_sigma
        self.reg_alpha = reg_alpha
        optimizer_fg_params = list(self.models['cebm_fg'].parameters())
        if full_grad:
            optimizer_fg_params += list(self.models['cebm_bg'].parameters())
        self.optimizer_fg = torch.optim.Adam(optimizer_fg_params, lr=lr)
        self.optimizer_bg = torch.optim.Adam(list(self.models['cebm_bg'].parameters()), lr=lr)
        self.metric_names = ['Eb_div', 'Eb_data', 'Eb_model', 'Ef_div', 'Ef_data', 'Ef_model', 'reg']
        self.disc_gamma = disc_gamma
        
    def train_epoch(self, epoch):
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images_fg, images_bg, _) in enumerate(self.train_loader):
            images_fg = (images_fg + self.img_sigma * torch.randn_like(images_fg)).to(self.device)
            images_bg = (images_bg + self.img_sigma * torch.randn_like(images_bg)).to(self.device)
            self.optimizer_bg.zero_grad() 
            loss_bg, metric_epoch = self.loss_bg(images_bg, metric_epoch)
            if metric_epoch['Eb_div'].abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
            loss_bg.backward()
            self.optimizer_bg.step()
            
            self.optimizer_fg.zero_grad()
            loss_fg, metric_epoch = self.loss_fg(images_fg, metric_epoch)
            if metric_epoch['Ef_div'].abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
            loss_fg.backward()
            self.optimizer_fg.step() 
#             for idx in range(self.train_schedule):
#                 self.optimizer_disc.zero_grad()
#                 self.optimizer_bg.zero_grad()
#                 self.optimizer_fg.zero_grad()
#                 loss_disc, metric_epoch = self.loss_disc(images_fg, metric_epoch, idx)
#                 loss_disc.backward()
#                 self.optimizer_disc.step()
#             break
        if epoch == (self.num_epochs-1):
            buffer = {'fg' : self.sgld_sampler_fg.buffer.cpu().detach(),
                      'bg' : self.sgld_sampler_bg.buffer.cpu().detach(),
                     }
            torch.save(buffer, "./weights/buffer-{:s}".format(self.exp_name))
        return {k: (v.item() / (b+1)) for k, v in metric_epoch.items()}

    def post_dists(self, images, joint=True):
        bg_mean, bg_std = self.models['cebm_bg'].latent_params(images)
        fg_mean, fg_std = self.models['cebm_fg'].latent_params(images)
#         return bg_mean, fg_mean
#         if joint:
#             return torch.cat((bg_mean, fg_mean), -1)
#         else:
#             return torch.cat((bg_mean[torch.randperm(bg_mean.shape[0])], fg_mean), -1)
#         bg_dist = torch.distributions.Normal(bg_mean, bg_std)
#         fg_dist = torch.distributions.Normal(fg_mean, fg_std)
#         return bg_dist, fg_dist
    
    def asampling(self, bg_dist, fg_dist, joint=True):
        bg_samples = bg_dist.rsample()
        fg_samples = fg_dist.rsample()
        if joint:
            return torch.cat((bg_samples, fg_samples), -1)
        else:
            return torch.cat((bg_samples[torch.randperm(bg_samples.shape[0])], fg_samples), -1)
        
    def loss_bg(self, data_images, metric_epoch):
        E_data = self.models['cebm_bg'].energy(data_images)
        sim_images = self.sgld_sampler_bg.sample(self.models['cebm_bg'].energy, len(data_images), self.sgld_steps)
        E_model = self.models['cebm_bg'].energy(sim_images)
        E_div = E_data.mean() - E_model.mean() 
        loss = E_div + self.reg_alpha * ((E_data**2).mean() + (E_model**2).mean())
        metric_epoch['Eb_div'] += E_div.detach()
        metric_epoch['Eb_data'] += E_data.mean().detach()
        metric_epoch['Eb_model'] += E_model.mean().detach()
        return loss, metric_epoch
    
    def fg_energy(self, images):
        return self.models['cebm_fg'].energy(images) + self.models['cebm_bg'].energy(images)
    
    def loss_fg(self, data_images, metric_epoch, EPS=1e-8):
        E_data = self.fg_energy(data_images)
        sim_images = self.sgld_sampler_fg.sample(self.fg_energy, len(data_images), self.sgld_steps)
        E_model = self.fg_energy(sim_images)
        E_div = E_data.mean() - E_model.mean() 
        loss = E_div + self.reg_alpha * ((E_data**2).mean() + (E_model**2).mean())
#         bg_dist, fg_dist = self.post_dists(data_images)
#         z_joint = self.asampling(bg_dist, fg_dist, joint=True)
#         z_joint = self.post_dists(data_images, joint=True)
        bg_mean, _ = self.models['cebm_bg'].latent_params(data_images)
        fg_mean, _ = self.models['cebm_fg'].latent_params(data_images)
#         reg = ((bg_mean * fg_mean).sum(-1)**2).mean()
        loss += self.disc_gamma * reg
#         D = self.models['disc'].pred_probs(z_joint)
#         loss += self.disc_gamma * ((D + EPS).log() - (1. - D + EPS).log()).mean()
        metric_epoch['reg'] += reg.detach()
        metric_epoch['Ef_div'] += E_div.detach()
        metric_epoch['Ef_data'] += E_data.mean().detach()
        metric_epoch['Ef_model'] += E_model.mean().detach()
        return loss, metric_epoch

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': True}
    train_loader, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
    
    network_args = {'device': device,
                    'im_height': im_h, 
                    'im_width': im_w, 
                    'input_channels': im_channels, 
                    'channels': eval(args.channels), 
                    'kernels': eval(args.kernels), 
                    'strides': eval(args.strides), 
                    'paddings': eval(args.paddings),
                    'hidden_dims': eval(args.hidden_dims),
                    'latent_dim': args.latent_dim,
                    'activation': args.activation,
                    'disc_input_dim': args.latent_dim*2,
                    'disc_hidden_dims': eval(args.disc_hidden_dims),
                    'disc_activation': 'ReLU',
                    }
    
    
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
                    'sgld_sampler_fg': SGLD_Sampler(**sgld_args),
                    'sgld_sampler_bg': SGLD_Sampler(**sgld_args),
                    'lr': args.lr,
                    'sgld_steps': args.sgld_steps,
                    'img_sigma': args.img_sigma,
                    'reg_alpha': args.reg_alpha,
                    'full_grad': args.full_grad,
                    'disc_gamma': args.disc_gamma}
    
    trainer = Train_GCEBM(**trainer_args)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='GCEBM', choices=['CVAECEBM', 'GCEBM'])
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--exp_id', default=None, type=str)
    ## data config
    parser.add_argument('--data', default='grassymnist_subgrass', choices=['grassymnist_grass', 'grassymnist_subgrass'])
    parser.add_argument('--img_sigma', default=3e-2, type=float)
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
    parser.add_argument('--disc_hidden_dims', default="[512,128,32]")
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    ## sgld sampler config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--reuse_freq', default=0.95, type=float)
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_alpha', default=2.0, type=float, help='step size is half of this value')
    parser.add_argument('--sgld_steps', default=60, type=int)
    parser.add_argument('--reg_alpha', default=5e-3, type=float)    
    parser.add_argument('--full_grad', default=False, action='store_true')
    parser.add_argument('--disc_gamma', default=1.0, type=float)
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)
