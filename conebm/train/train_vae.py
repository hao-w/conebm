import os
import torch
import argparse
from sebm.sgld import SGLD_Sampler
from sebm.data import setup_data_loader
from sebm.utils import set_seed, create_exp_name, init_models, save_models, Trainer
        
class Train_VAE(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, lr, sample_size):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.optimizer = torch.optim.Adam(list(self.models['enc'].parameters())+list(self.models['dec'].parameters()), lr=lr)
        self.metric_names = ['elbo', 'll', 'kl']
        self.sample_size = sample_size

    def train_epoch(self, epoch):
        enc = self.models['enc']
        dec = self.models['dec']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images = images.repeat(self.sample_size, 1, 1, 1, 1).to(self.device)
            loss, metric_epoch = self.loss(enc, dec, images, metric_epoch)
            loss.backward()
            self.optimizer.step()
        return {k: (v.item() / (b+1)) for k, v in metric_epoch.items()}
    
    def loss(self, enc, dec, x, metric_epoch):
        z, log_qz = enc.forward(x)
        recon, ll, log_pz = dec.forward(z, x)
        kl = log_qz - log_pz
        elbo = (ll - kl).mean()
        metric_epoch['elbo'] += elbo.detach()
        metric_epoch['ll'] += ll.mean().detach()
        metric_epoch['kl'] += kl.mean().detach()
        return - elbo, metric_epoch    
    
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True if not args.no_digit else None, 
                    'normalize': False}
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
                    'dec_paddings': eval(args.dec_paddings)}
    
    models = init_models(args.model_name, device, network_args)
    
    print('Start Training..')
    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'lr': args.lr,
                    'sample_size': args.sample_size}
    trainer = Train_VAE(**trainer_args)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='VAE', choices=['VAE'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--exp_id', default=None, type=str)
    ## data config
    parser.add_argument('--data', default='grassymnist', choices=['mnist', 'flowermnist', 'grassymnist'])
    parser.add_argument('--no_digit', default=False, action='store_true')
    ## optim config
    parser.add_argument('--lr', default=1e-3, type=float)
    ## arch config 
    parser.add_argument('--channels', default="[32,32,64,64]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--dec_paddings', default="[1,1,0,0]")
    parser.add_argument('--hidden_dims', default="[256]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='ReLU')
    parser.add_argument('--leak_slope', default=0.1, type=float, help='parameter for LeakyReLU activation')
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--sample_size', default=1, type=int)
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)
