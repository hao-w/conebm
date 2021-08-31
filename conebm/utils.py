import os
import git
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PARENT_DIR = git.Repo(os.path.abspath(os.curdir), search_parent_directories=True).git.rev_parse("--show-toplevel")

def plot_samples(images, denormalize, fs=.5, save_name=None):
    test_batch_size = len(images)
    images = images.squeeze().cpu().detach()
    images = torch.clamp(images, min=-1, max=1)
    if denormalize:
        images = images * 0.5 + 0.5
    gs = gridspec.GridSpec(int(test_batch_size/10), 10)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.1, hspace=0.1)
    fig = plt.figure(figsize=(fs*10, fs*int(test_batch_size/10)))
    for i in range(test_batch_size):
        ax = fig.add_subplot(gs[int(i/10), i%10])
        try:
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
        except:
            ax.imshow(np.transpose(images[i], (1,2,0)), vmin=0, vmax=1.0)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
    result_path = os.path.join(PARENT_DIR, 'sebm', 'results')
    if save_name is not None:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        plt.savefig(os.path.join(result_path, 'samples_{}.png'.format(save_name)), dpi=300)
        
def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def create_exp_name(args): 
    if args.model_name in ['CEBM', 'GCEBM', 'SUPCEBM', 'GCEBM-restore']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_sgld_s=%s_n=%s_a=%s_dn=%s_reg=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation,
                     args.sgld_steps, args.sgld_noise_std, args.sgld_alpha, args.img_sigma, 
                     args.reg_alpha, args.seed)
            
        if args.model_name in ['GCEBM', 'GCEBM-restore', 'SUPCEBM']:
            exp_name += '_gamma={}'.format(args.disc_gamma)
            
        if args.model_name in ['GCEBM']:
            exp_name += '_{}'.format(args.optim_method)
            
    elif args.model_name in ['CEBM-SSM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation, args.seed)
        
    elif args.model_name in ['VAE', 'SUPVAE']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation, args.seed)
        
    elif args.model_name in ['VERA']:
        exp_name = '%s_d=%s_z=%s_lrp=%s_lrq=%s_lrx=%s_gamma=%s_lamb=%s' % (args.model_name, args.data, args.latent_dim, args.lr_p, args.lr_q, args.lr_xee, args.gamma, args.lambda_ent)

    else:
        raise ValueError
        
    if args.exp_id is not None:
        exp_name += '_{}'.format(args.exp_id)

        
    print("Experiment: %s" % exp_name)
    return exp_name

def init_models(model_name, device, network_args):  
    if model_name in ['CEBM-SSM', 'CEBM', 'CCEBM', 'CEBM-newnss']:
        from sebm.model.cebm import CEBM_Gaussian
        cebm = CEBM_Gaussian(**network_args).to(device)      
        return {'cebm': cebm}
    
    if model_name in ['SUPCEBM','Train_NCECEBM']:
        from sebm.model.cebm import SUPCEBM
        cebm = SUPCEBM(**network_args).to(device)      
        return {'cebm': cebm}
    
    elif model_name in ['GCEBM', 'GCEBM-restore']:
        from sebm.model.cebm import CEBM_Gaussian
        cebm_fg = CEBM_Gaussian(**network_args).to(device)      
        cebm_bg = CEBM_Gaussian(**network_args).to(device)
        return {'cebm_fg': cebm_fg, 'cebm_bg': cebm_bg}

    elif model_name in ['VAE']:
        from sebm.model.vae import Encoder, Decoder_Gaussian
        enc = Encoder(**network_args).to(device)
        dec = Decoder_Gaussian(**network_args).to(device)
        return {'enc': enc, 'dec': dec}
    
    elif model_name in ['SUPVAE']:
        from sebm.model.vae import Encoder, Decoder_Gaussian
        enc = SUPEncoder(**network_args).to(device)
        dec = SUPDecoder_Gaussian(**network_args).to(device)
        return {'enc': enc, 'dec': dec}    
    
    elif model_name in ['VERA']:
        from sebm.model.cebm import CEBM_Gaussian, VERA_Generator, VERA_Xee
        model = CEBM_Gaussian(**network_args) 
        gen = VERA_Generator(**network_args)
        xee = VERA_Xee(device, network_args['latent_dim'], network_args['xee_init_sigma'])
        return {'cebm': model.to(device), 'gen': gen.to(device), 'xee': xee}
    
    else:
        raise ValueError()
    
def save_models(models, filename, weights_dir=os.path.join(PARENT_DIR, "sebm", "weights")):
    checkpoint = {k: v.state_dict() for k, v in models.items()}
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(models, filename, weights_dir=os.path.join(PARENT_DIR, "sebm", "weights"), **kwargs):
    checkpoint = torch.load(f'{weights_dir}/{filename}', **kwargs)
    {k: v.load_state_dict(checkpoint[k]) for k, v in models.items()}  

def print_path(path = os.path.join(PARENT_DIR, 'sebm', 'test_folder')):
    print(path)
    
class Trainer():
    """
    A generic model trainer
    """
    def __init__(self, models, train_loader, num_epochs, device, exp_name):
        super().__init__()
        self.models = models
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.exp_name = exp_name
        self.logging_path = os.path.join(PARENT_DIR, "sebm", "logging")
    def train_epoch(self, epoch):
        pass
    
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:            
            metric_epoch = self.train_epoch(epoch)
            pbar.set_postfix(ordered_dict=metric_epoch)
            self.logging(metric_epoch, epoch)
            save_models(self.models, self.exp_name)
    
    #FIXME: hao will replace this function with tensorboard API later.
    
    def logging(self, metrics, epoch):
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path)
        fout = open(os.path.join(self.logging_path, self.exp_name + '.txt'), mode='w+' if epoch==0 else 'a+')
        metric_print = ",  ".join(['{:s}={:.2e}'.format(k, v) for k, v in metrics.items()])
        print("Epoch={:d}, ".format(epoch+1) + metric_print, file=fout)
        fout.close()