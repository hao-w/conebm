import os
import torch
import argparse
from sebm.sgld import SGLD_Sampler
from sebm.data import setup_data_loader
from sebm.utils import set_seed, create_exp_name, init_models, save_models, Trainer


        
class Train_CEBM(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, lr, img_sigma, reg_alpha):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.img_sigma = img_sigma
        self.reg_alpha = reg_alpha
        self.optimizer = torch.optim.Adam(list(self.models['cebm'].parameters()), lr=lr)
        self.metric_names = ['Loss', 'E_data']

    def train_epoch(self, epoch):
        ebm = self.models['cebm']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad() 
#             images = (images + self.img_sigma * torch.randn_like(images)).to(self.device)
            images = images.to(self.device)
            loss, metric_epoch = self.loss_ssm(ebm, images, metric_epoch)
            loss.backward()
            self.optimizer.step()
#             break
        return {k: (v.item() / (b+1)) for k, v in metric_epoch.items()}

    def loss_ssm(self, ebm, data_images, metric_epoch):
        """
        sliced score matching
        """
        data_images.requires_grad = True
        E_data = ebm.energy(data_images)
        v = torch.randn_like(data_images)
        grads = torch.autograd.grad(outputs=E_data.sum(), inputs=data_images, create_graph=True, retain_graph=True)[0]
        sq_sum_first_derivative = (grads**2).sum(-1) * 0.5
        second_grads = torch.autograd.grad(outputs=(grads * v).sum(-1).sum(), inputs=data_images, retain_graph=True)[0]
        sum_second_derivative = (second_grads * v).sum(-1)
        loss = (sq_sum_first_derivative + sum_second_derivative).mean() 
#         loss = E_div + self.reg_alpha * ((E_data**2).mean() + (E_model**2).mean())
        metric_epoch['Loss'] += loss.detach()
        metric_epoch['E_data'] += E_data.mean().detach()
        return loss, metric_epoch
    
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'batch_size': args.batch_size,
                    'train': True, 
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
                    'activation': args.activation}
    
    models = init_models(args.model_name, device, network_args)
    
    print('Start Training..')

    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'lr': args.lr,
                    'img_sigma': args.img_sigma,
                    'reg_alpha': args.reg_alpha}
    trainer = Train_CEBM(**trainer_args)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='CEBM-SSM', choices=['CEBM-SSM', 'CEBM', 'GCEBM'])
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--exp_id', default=None, type=str)
    ## data config
    parser.add_argument('--data', default='mnist', choices=['mnist', 'flowermnist', 'grassymnist'])
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
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    ## sgld sampler config
    parser.add_argument('--reg_alpha', default=5e-3, type=float)    
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)
