import os
import umap
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sebm.data import setup_data_loader
from sebm.utils import load_models, set_seed
from sebm.sgld import SGLD_Sampler
             

def generate_samples(self, batch_size, **kwargs):
    if self.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
        raise NotImplementedError()
  
    elif self.model_name in ['VAE', 'VAE_GMM', 'BIGAN', 'BIGAN_GMM']:
        return self.models['enc'].latent_params()
    else:
        raise NotImplementError

class Evaluator():
    def __init__(self, device, models, model_name, data, data_dir='../data/', **kwargs):
        super().__init__()
        self.models = models
        self.model_name = model_name
        self.data = data
        self.data_dir = data_dir
        self.device = device
        
    def latent_mode(self, images):
        if self.model_name in ['CEBM', 'VERA', 'CEBM-newnss']:
            return self.models['cebm'].latent_params(images)
        elif self.model_name in ['GCEBM', 'GCEBM-NOREG']:
            return self.models['cebm_fg'].latent_params(images)
        elif self.model_name in ['SUPCEBM']:
            labels = torch.nn.functional.one_hot(torch.ones(len(images), device=images.device, dtype=torch.long), num_classes=2)
            return self.models['cebm'].latent_params(images, labels)
        elif self.model_name in ['VAE']:
            return self.models['enc'].latent_params(images)
        else:
            raise NotImplementError
            
    def encode_dataset(self, data_loader):
        """
        return the modes of latent representations with the correspoding labels
        """
        zs = []
        ys = []
        for b, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            if self.model_name in ['VAE', 'VAE_GMM']:
                images = images.unsqueeze(0)
            mean, _ = self.latent_mode(images)
            zs.append(mean.squeeze(0).detach().cpu().numpy())
            ys.append(labels.numpy())
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys
    
    def project_latent(self, algorithm='umap'):
        test_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                 data_dir=self.data_dir, 
                                                                 batch_size=1000, 
                                                                 train=False, 
                                                                 normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                 shuffle=False, 
                                                                 shot_random_seed=None)

        zs, ys = self.encode_dataset(test_loader)
#         breakpoint()
        if algorithm == 'umap':
            reducer = umap.UMAP()
        elif algorithm == 'tsne':
            reducer = TSNE(n_components=2)
        zs2 = reducer.fit_transform(zs)
        print('{} projection completed, visualizing 2D space..'.format(algorithm))
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        colors = []
        for k in range(10):
            m = (ys == k)
            p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=5)
            colors.append(p.get_facecolor())
        ax.legend()
        fig.tight_layout()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        plt.savefig('results/{}_d={}_{}'.format(algorithm, self.data, self.model_name), dpi=300)
    
    def few_label_classification(self, list_num_shots=[1, 10, 100, -1], num_runs=10, batch_size=1000, classifier='logistic'):
        """
        Import classifier implementations from scikit-learn  
        train logistic classifiers with the encoded representations of the training set
        compute the accuracy for the test set
        """
        if not os.path.exists('results/few_label/'):
            os.makedirs('results/few_label/')
        results = {'Num_Shots': [], 'Mean': [], 'Std': []}
        train_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                  data_dir=self.data_dir, 
                                                                  batch_size=batch_size, 
                                                                  train=True, 
                                                                  normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                  shuffle=False, 
                                                                  shot_random_seed=None)
        zs_train, ys_train = self.encode_dataset(train_loader)
        
        test_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                 data_dir=self.data_dir, 
                                                                 batch_size=batch_size, 
                                                                 train=False, 
                                                                 normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                 shuffle=False, 
                                                                 shot_random_seed=None)
        zs_test, ys_test = self.encode_dataset(test_loader)
        
        num_classes = len(np.unique(ys_train))
        
        for num_shots in tqdm(list_num_shots):
            Accuracy = []
            if num_shots == -1:
                clf = LogisticRegression(random_state=0, 
                                         multi_class='auto', 
                                         solver='liblinear', 
                                         max_iter=10000).fit(zs_train, ys_train)
                Accuracy.append(np.array([clf.score(zs_test, ys_test)]))                
            else:
                for i in tqdm(range(num_runs)):
    #                 torch.cuda.empty_cache()
                    zs_train_selected = []
                    ys_train_selected = []
                    set_seed(i)
                    for k in range(num_classes):
                        indk = np.argwhere(ys_train == k)[:,0]
                        ind_of_indk = np.random.permutation(len(indk))[:num_shots]
                        indk_selected = indk[ind_of_indk]
                        zs_train_selected.append(zs_train[indk_selected])
                        ys_train_selected.append(ys_train[indk_selected])
                    zs_train_selected = np.concatenate(zs_train_selected, 0)
                    ys_train_selected = np.concatenate(ys_train_selected, 0)

#                     print(indk_selected)
                    
                    clf = LogisticRegression(random_state=0, 
                                             multi_class='auto', 
                                             solver='liblinear', 
                                             max_iter=10000).fit(zs_train_selected, ys_train_selected)
                    Accuracy.append(np.array([clf.score(zs_test, ys_test)]))  
#                 return 0
            Accuracy = np.concatenate(Accuracy)
            results['Num_Shots'].append(num_shots)
            results['Mean'].append(Accuracy.mean())
            results['Std'].append(Accuracy.std())
        pd.DataFrame.from_dict(results).to_csv('results/few_label/{}-{}-runs={}.csv'.format(self.model_name, self.data, num_runs), index=False)
        return results
