import time
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class SGLD_Sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, im_h, im_w, im_channels, device, alpha, noise_std, buffer_size, reuse_freq):
        super().__init__()
        im_dims = (im_channels, im_h, im_w)
        self.initial_dist = Uniform(-1 * torch.ones(im_dims).to(device), torch.ones(im_dims).to(device))
        self.device = device
        self.alpha = alpha
        self.noise_std = noise_std
        self.reuse_freq = reuse_freq
        self.buffer = self.initial_dist.sample((buffer_size, ))

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        samples = self.initial_dist.sample((batch_size, ))
        inds = torch.randint(0, len(self.buffer), (batch_size, ), device=self.device)
        samples_from_buffer = self.buffer[inds]
        rand_mask = (torch.rand(batch_size, device=self.device) < self.reuse_freq)
        samples[rand_mask] = samples_from_buffer[rand_mask]
        return samples, inds
    
    def refine_buffer(self, samples, inds):
        """
        update replay buffer
        """
        self.buffer[inds] = samples

    def sample(self, energy, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using sgld
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=energy(samples).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            samples = samples.detach() 
#         samples = torch.tanh(samples)
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
    
    def cond_sample(self, ebm, y, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.cond_energy(samples, y).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            #added this extra detachment step, becase the last update keeps the variable in the graph.
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
    

class SGLD_Sampler2():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, im_h, im_w, im_channels, device, alpha, noise_std, buffer_size, reuse_freq):
        super().__init__()
        im_dims = (im_channels, im_h, im_w)
        self.initial_dist = Uniform(-1 * torch.ones(im_dims).to(device), torch.ones(im_dims).to(device))
        self.device = device
        self.alpha = alpha
        self.noise_std = noise_std
        self.reuse_freq = reuse_freq
        self.buffer_fg = self.initial_dist.sample((buffer_size, ))
        self.buffer_bg = self.initial_dist.sample((buffer_size, ))

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        actual_batch_size = int(batch_size/2)
        samples_fg = self.initial_dist.sample((actual_batch_size, ))
        samples_bg = self.initial_dist.sample((actual_batch_size, ))

        inds_fg = torch.randint(0, len(self.buffer_fg), (actual_batch_size, ), device=self.device)
        inds_bg = torch.randint(0, len(self.buffer_bg), (actual_batch_size, ), device=self.device)

        samples_from_buffer_fg = self.buffer_fg[inds_fg]
        samples_from_buffer_bg = self.buffer_bg[inds_bg]
        
        rand_mask_fg = (torch.rand(actual_batch_size, device=self.device) < self.reuse_freq)
        rand_mask_bg = (torch.rand(actual_batch_size, device=self.device) < self.reuse_freq)

        samples_fg[rand_mask_fg] = samples_from_buffer_fg[rand_mask_fg]
        samples_bg[rand_mask_bg] = samples_from_buffer_bg[rand_mask_bg]
        
        return samples_fg, samples_bg, inds_fg, inds_bg
    
    def refine_buffer(self, samples_fg, samples_bg, inds_fg, inds_bg):
        self.buffer_fg[inds_fg] = samples_fg
        self.buffer_bg[inds_bg] = samples_bg

    def sample(self, fg_energy, bg_energy, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples_fg, samples_bg, inds_fg, inds_bg = self.sample_from_buffer(batch_size)
            for l in range(num_steps):
                samples_fg.requires_grad = True
                samples_bg.requires_grad = True
                grads = torch.autograd.grad(outputs=fg_energy(samples_fg).sum()+bg_energy(samples_bg).sum(), 
                                            inputs=[samples_fg, samples_bg])
                
                samples_fg = (samples_fg - (self.alpha / 2) * grads[0] + self.noise_std * torch.randn_like(grads[0])).detach()
                samples_bg = (samples_bg - (self.alpha / 2) * grads[1] + self.noise_std * torch.randn_like(grads[1])).detach()

                #added this extra detachment step, becase the last update keeps the variable in the graph.
                samples_fg = samples_fg.detach() 
                samples_bg = samples_bg.detach()
#             assert samples.requires_grad == False, "samples should not require gradient.
                self.refine_buffer(samples_fg.detach(), samples_bg.detach(), inds_fg, inds_bg)
            return samples_fg, samples_bg
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples

            list_samples = []
            for l in range(num_steps):
                samples.requires_grad = True
                grads = torch.autograd.grad(outputs=energy(samples).sum(), inputs=samples)[0]
                samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
                #added this extra detachment step, becase the last update keeps the variable in the graph.
                samples = samples.detach() 
            assert samples.requires_grad == False, "samples should not require gradient."
            if pcd:
                self.refine_buffer(samples.detach(), inds_fg, inds_bg)
            return samples
    
    def cond_sample(self, ebm, y, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples_fg, samples_bg, inds_fg, inds_bg = self.sample_from_buffer(batch_size)
            samples = torch.cat((samples_fg, samples_bg), 0)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.cond_energy(samples, y).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            actual_batch_size = int(len(samples) / 2)
            self.refine_buffer(samples[:actual_batch_size].detach(), samples[actual_batch_size:], inds_fg, inds_bg)
        return samples
    
