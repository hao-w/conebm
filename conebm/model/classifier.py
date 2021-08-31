import torch
import torch.nn as nn
from sebm.net import cnn_block, mlp_block

class Discriminator(nn.Module):
    """
    A discriminator that tries to disentangle the latent features
    """
    def __init__(self, input_dim, hidden_dims, activation, **kwargs):
        super().__init__() 
        self.mlp_net = mlp_block(input_dim, hidden_dims, activation, **kwargs)
        self.output_net = nn.Sequential(
                            nn.Linear(hidden_dims[-1], 1),
                            nn.Sigmoid())
        self.bceloss = nn.BCELoss()
        
    def pred_probs(self, x):
        probs = self.output_net(self.mlp_net(x))
        assert probs.shape == (x.shape[0], 1)
        return probs.squeeze(-1)
        
    def loss(self, pred_probs, labels):
        return self.bceloss(pred_probs, labels)
    
    def score(self, pred_probs, labels, joint=True):
        binary_pred_probs = torch.zeros(pred_probs.shape, device=pred_probs.device)
        binary_pred_probs[(pred_probs > 0.5)] = 1
        return (binary_pred_probs == labels).float().sum()
    
class SEMI_CLF(nn.Module):
    """
    a (semi-)supervised classifier
    """
    def __init__(self, **kwargs):
        super().__init__() 
        self.clf_net = nn.Sequential(*(list(cnn_mlp_1out(**kwargs)) + [nn.LogSoftmax(dim=-1)]))
        self.nllloss = nn.NLLLoss()
        
    def pred_logit(self, images):
        return self.clf_net(images)
    
    def loss(self, pred_logits, labels):
        return self.nllloss(pred_logits, labels)
    
    def score(self, images, labels):
        pred_logits = self.pred_logit(images)
        return (pred_logits.argmax(-1) == labels).float().sum()