import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from sebm.net import cnn_block, deconv_block, mlp_block, cnn_output_shape, Reshape
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat


####################################################
#exponential family term of a Gaussian distribution#
####################################################
def log_partition(nat1, nat2):
    """
    compute the log partition of a normal distribution
    """
    return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  

def nats_to_params(nat1, nat2):
    """
    convert a Gaussian natural parameters its distritbuion parameters,
    mu = - 0.5 *  (nat1 / nat2), 
    sigma = (- 0.5 / nat2).sqrt()
    nat1 : natural parameter which correspond to x,
    nat2 : natural parameter which correspond to x^2.      
    """
    mu = - 0.5 * nat1 / nat2
    sigma = (- 0.5 / nat2).sqrt()
    return mu, sigma

def params_to_nats(mu, sigma):
    """
    convert a Gaussian distribution parameters to the natrual parameters
    nat1 = mean / sigma**2, 
    nat2 = - 1 / (2 * sigma**2)
    nat1 : natural parameter which correspond to x,
    nat2 : natural parameter which correspond to x^2.
    """
    nat1 = mu / (sigma**2)
    nat2 = - 0.5 / (sigma**2)
    return nat1, nat2  
    
class CEBM_Gaussian(nn.Module):
    """
    A image-level energy-based encoder 
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super(CEBM_Gaussian, self).__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mlp_net  = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.nss1_net = nn.Linear(hidden_dims[-1], latent_dim)
        self.nss2_net = nn.Linear(hidden_dims[-1], latent_dim)
             
        self.ib_mean = torch.zeros(latent_dim, device=self.device)
        self.ib_log_std = torch.zeros(latent_dim, device=self.device)
        
    def forward(self, x):
        h = self.mlp_net(self.flatten(self.conv_net(x)))
        nss1 = self.nss1_net(h) 
        nss2 = self.nss2_net(h)
#        return nss1, - (nss2 - nss1)**2
        return nss1, - nss2**2

    def log_factor(self, nss1, nss2, latents, expand_dim=False):
        if expand_dim:
            return (nss1[None, :, :] * latents[:, None, :]).sum(2) \
                    + (nss2[None, :, :]  * (latents[:, None, :]**2)).sum(2)
        else:
            return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 
    
    def energy(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = log_partition(ib_nat1, ib_nat2)
        logA_posterior = log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def energy_p2(self, nss1, nss2):
        ib_nat1, ib_nat2 = params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = log_partition(ib_nat1, ib_nat2)
        logA_posterior = log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1) 
    
    def latent_params(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
        return nats_to_params(ib_nat1+nss1, ib_nat2+nss2)
    
    def log_prior(self, latents):
        return Normal(self.ib_mean, self.ib_log_std.exp()).log_prob(latents).sum(-1)      
    
class SUPCEBM(nn.Module):
    """
    A image-level energy-based encoder 
    """
    def __init__(self, device, num_classes, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super(SUPCEBM, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mlp_net  = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.nss1_net = nn.Linear(hidden_dims[-1]+num_classes, latent_dim)
        self.nss2_net = nn.Linear(hidden_dims[-1]+num_classes, latent_dim)
             
        self.ib_mean = torch.zeros(latent_dim, device=self.device)
        self.ib_log_std = torch.zeros(latent_dim, device=self.device)
        
    def forward(self, x, y):
        h1 = self.mlp_net(self.flatten(self.conv_net(x)))
        h2 = torch.cat((h1, y), -1)
        nss1 = self.nss1_net(h2) 
        nss2 = self.nss2_net(h2)
        return nss1, - nss2**2

    def log_factor(self, x, y, latents, expand_dim=None):
        nss1, nss2 = self.forward(x, y)
        if expand_dim is not None:
            nss1 = nss1.expand(expand_dim , -1, -1)
            nss2 = nss2.expand(expand_dim , -1, -1)
            return (nss1 * latents).sum(2) + (nss2 * (latents**2)).sum(2)
        else:
            return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 
    
    def energy(self, x):
        Energy = []
        for i in range(self.num_classes):
            y = torch.nn.functional.one_hot(torch.ones(len(x)).long()*i, num_classes=self.num_classes).to(self.device)
            cond_energy = self.cond_energy(x, y)
            assert cond_energy.shape == (len(x),)
            Energy.append(cond_energy)
        Energy = torch.stack(Energy, -1)
        y_post_probs = torch.nn.functional.softmax(-Energy, dim=-1).detach().argmax(-1)
        return - torch.logsumexp(-Energy, dim=-1), y_post_probs

    def cond_energy(self, x, y):
        nss1, nss2 = self.forward(x, y)
        ib_nat1, ib_nat2 = params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = log_partition(ib_nat1, ib_nat2)
        logA_posterior = log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1) 
    
    def latent_params(self, x, y):
#         _, y_post_probs = self.energy(x)
#         y = torch.nn.functional.one_hot(y_post_probs, num_classes=self.num_classes)
        nss1, nss2 = self.forward(x, y)
        ib_nat1, ib_nat2 = params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
        return nats_to_params(ib_nat1+nss1, ib_nat2+nss2) 
    
    def log_prior(self, latents):
        return Normal(self.ib_mean, self.ib_log_std.exp()).log_prob(latents).sum(-1)      
    
    
class VERA_Generator(nn.Module):
    def __init__(self, device, gen_channels, gen_kernels, gen_strides, gen_paddings, latent_dim, gen_activation, likelihood='gaussian', **kwargs):
        super().__init__()
        self.gen_net = deconv_block(im_height=1, im_width=1, input_channels=latent_dim, channels=gen_channels, kernels=gen_kernels, strides=gen_strides, paddings=gen_paddings, activation=gen_activation, last_act=False, batchnorm=True)
        self.likelihood = likelihood
        self.device = device
        self.latent_dim = latent_dim  
        self.prior_mean = torch.zeros(latent_dim, device=self.device)
        self.prior_log_std = torch.zeros(latent_dim, device=self.device)

        if self.likelihood == 'gaussian':
            self.last_act = nn.Tanh()
            self.x_logsigma = nn.Parameter((torch.ones(1, device=self.device) * .01).log())

        elif self.likelihood == 'cb':
            self.last_act = nn.Sigmoid()

    def forward(self, z):
        return self.last_act(self.gen_net(z[..., None, None]))

    def sample(self, batch_size):
        z0 = Normal(self.prior_mean, self.prior_log_std.exp()).sample((batch_size,))
        xr_mu = self.last_act(self.gen_net(z0[..., None, None]))
        if self.likelihood == 'gaussian':
            xr = xr_mu + torch.randn_like(xr_mu) * self.x_logsigma.exp()
        elif self.likelihood == 'cb':
            xr = CB(probs=xr_mu).rsample()
        return z0, xr, xr_mu
        
    def log_joint(self, x, z):
        log_p_z = Normal(self.prior_mean, self.prior_log_std.exp()).log_prob(z).sum(-1)
        if z.dim() == 2:
            x_mu = self.last_act(self.gen_net(z[..., None, None]))
            if self.likelihood == 'gaussian':
                ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x).sum(-1).sum(-1).sum(-1)
            elif self.likelihood == 'cb':
                ll = CB(probs=x_mu).log_prob(x).sum(-1).sum(-1).sum(-1)
        elif z.dim() == 3:
            S, B, D = z.shape
            x_mu = self.last_act(self.gen_net(z.view(S*B, -1)[..., None, None]))
            x_mu = x_mu.view(S, B, *x_mu.shape[1:])
            if self.likelihood == 'gaussian':
                ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x[None]).sum(-1).sum(-1).sum(-1)
            elif self.likelihood == 'cb':
                ll = CB(probs=x_mu).log_prob(x[None]).sum(-1).sum(-1).sum(-1)
        assert ll.shape == log_p_z.shape
        return ll + log_p_z, x_mu
        
    def ll(self, x, z):
        if z.dim() == 2:
            x_mu = self.last_act(self.gen_net(z[..., None, None]))
            if self.likelihood == 'gaussian':
                ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x).sum(-1).sum(-1).sum(-1)
            elif self.likelihood == 'cb':
                ll = CB(probs=x_mu).log_prob(x).sum(-1).sum(-1).sum(-1)      
        elif z.dim() == 3:
            S, B, D = z.shape
            x_mu = self.last_act(self.gen_net(z.view(S*B, -1)[..., None, None]))
            x_mu = x_mu.view(S, B, *x_mu.shape[1:])
            if self.likelihood == 'gaussian':
                ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x[None]).sum(-1).sum(-1).sum(-1)
            elif self.likelihood == 'cb':
                ll = CB(probs=x_mu).log_prob(x[None]).sum(-1).sum(-1).sum(-1)                
        return ll

class VERA_Xee(nn.Module):
    def __init__(self, device, latent_dim, init_sigma):
        super().__init__()
        self.xee_logsigma = nn.Parameter((torch.ones(latent_dim, device=device) * init_sigma).log())
        
    def sample(self, z0, sample_size=None, detach_sigma=False, entropy=False):
        if detach_sigma:
            xee_dist = Normal(z0, self.xee_logsigma.exp().detach())
        else:
            xee_dist = Normal(z0, self.xee_logsigma.exp())
            
        if sample_size is None:
            z = xee_dist.rsample()
        else:
            z = xee_dist.rsample((sample_size,))
            
        if entropy:
            ent = xee_dist.entropy().sum(-1)
            return z, ent
        else:
            lp = xee_dist.log_prob(z).sum(-1)
            return z, lp
