import torch
from sebm.data import setup_data_loader
from sebm.utils import load_models, init_models, plot_samples
from sebm.sgld import SGLD_Sampler
from sebm.eval import Evaluator

model_name = 'GCEBM' # "VAE", "CEBM"
data =  'grassymnist_grass' #'mnist'
dataset_args = {'data': data, 
                'batch_size': 100,
                'train': True, 
                'normalize': False if model_name=='VAE' else True,
                'shuffle': False,}
_, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
num_classes = 2

device = 'cuda:0'
lr = 5e-5
reg_alpha = 5e-3
seed = 2
gamma = 10.0
latent_dim = 128
network_args = {'device': device,
                'im_height': im_h, 
                'im_width': im_w, 
                'input_channels': im_channels, 
                'channels': [32,32,64,64], 
                'kernels': [3,4,4,4], 
                'strides': [1,2,2,2], 
                'paddings': [1,1,1,1],
                'hidden_dims': [256],
                'latent_dim': 128,}

if model_name in ['GCEBM', 'CEBM', 'SUPCEBM']:
    network_args['activation'] = 'SiLU'
    if model_name == 'SUPCEBM':
        network_args['num_classes'] = num_classes

    exp_name = '{}_d={}_z={}_lr={}_act=SiLU_sgld_s=60_n=0.0075_a=2.0_dn=0.03_reg={}_seed={}'.format(model_name, data, latent_dim, lr, reg_alpha, seed)
    if model_name in ['GCEBM', 'SUPCEBM']:
        exp_name += '_gamma={}_hacky'.format(gamma)

elif model_name in ['VAE']:
    exp_name = '{}_d={}_z=128_lr=0.001_act=ReLU_seed={}'.format(model_name, data, seed)
    network_args['activation'] = 'ReLU'
    network_args['dec_paddings'] = [1,1,0,0]

models = init_models(model_name, device, network_args)
load_models(models, exp_name, map_location=torch.device(device))

evaluator = Evaluator(device, models, model_name, data='grassymnist')
evaluator.few_label_classification(batch_size=500)
