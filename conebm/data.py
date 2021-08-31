import os
import git
import torch
import tarfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join
from scipy.io import loadmat
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, ConcatDataset

PARENT_DIR = git.Repo(os.path.abspath(os.curdir), search_parent_directories=True).git.rev_parse("--show-toplevel")


def setup_data_loader(data, batch_size, train, normalize, shuffle=True, shot_random_seed=None, data_dir=os.path.join(PARENT_DIR, "data")):
    """
    This method constructs a dataloader for several commonly-used image datasets
    arguments:
        data: name of the dataset
        data_dir: path of the dataset
        num_shots: if positive integer, it means the number of labeled examples per class
                   if value is -1, it means the full dataset is labeled
        batch_size: batch size 
        train: if True return the training set;if False, return the test set
        normalize: if True, rescale the pixel values to [-1, 1]
        shot_random_seed: the random seed used when num_shot != -1
    """
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)
        
    if data in ['mnist', 'fashionmnist', 'flowermnist', 'grassymnist', 'grassymnist_grass', 'grassymnist_mnist','grassymnist_subgrass']:
        img_h, img_w, n_channels = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        img_h, img_w, n_channels = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((32,32)),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
    if not normalize:
        del transform.transforms[-1]

    if data == 'mnist':
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
    elif data == 'fashionmnist':
        dataset = datasets.FashionMNIST(data_dir, train=train, transform=transform, download=True)
    elif data == 'cifar10':
        data_dir = join(data_dir, 'CIFAR10')
        dataset = datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)
    elif data == 'svhn':
        data_dir = join(data_dir, 'SVHN')
        if train:
            train_part = datasets.SVHN(data_dir, split='train', transform=transform, download=True)
            extra_part = datasets.SVHN(data_dir, split='extra', transform=transform, download=True)
            dataset = ConcatDataset([train_part, extra_part])
        else:
            dataset = datasets.SVHN(data_dir, split='test', transform=transform, download=True)
    elif data == 'celeba':
        dataset = datasets.CelebA(data_dir, split='train', transform=transform, download=True, target_type='attr')

    elif data == 'grassymnist':
        if train is None:
            split = 'background'
        elif train:
            split = 'train'
        else:
            split = 'test'
        dataset = GrassyMNIST(data_dir, split=split, transform=transform, download=True)
        
    elif data == 'grassymnist_grass':
        if train:
            split = 'train'
        else:
            split = 'test'
        fg_dataset = GrassyMNIST(data_dir, split=split, transform=transform, download=True)
        bg_dataset = GrassyMNIST(data_dir, split='background', transform=transform, download=True)
        dataset = GrassyMNIST_both(fg_dataset, bg_dataset)
        
    elif data == 'grassymnist_subgrass':
        if train:
            split = 'train'
        else:
            split = 'test'
        fg_dataset = GrassyMNIST(data_dir, split=split, transform=transform, download=True)
        bg_dataset = GrassyMNIST(data_dir, split='background', transform=transform, download=True)
        labels = fg_dataset.targets
        fg_data = fg_dataset.data.clone()
        bg_data = bg_dataset.data.clone()
        bg_labels = bg_dataset.targets.clone()
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)
        idx = []
        for k in range(2):
            idxk = torch.where(labels == k)[0]
            idx.append(idxk)
        idx = torch.cat(idx, 0)
        fg_dataset.targets = labels[idx]
        fg_dataset.data = fg_data[idx]
        bg_dataset.data = bg_data[idx]
        bg_dataset.targets = bg_labels[idx]
        dataset = GrassyMNIST_both(fg_dataset, bg_dataset)    
        
    elif data == 'grassymnist_mnist':
        if train:
            split = 'train'
        else:
            split = 'test'
        fg_dataset = GrassyMNIST(data_dir, split=split, transform=transform, download=True)
        bg_dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        dataset = GrassyMNIST_both(fg_dataset, bg_dataset)
            
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            drop_last=True)
    return dataloader, img_h, img_w, n_channels

        
class GrassyMNIST_both(VisionDataset):
    def __init__(self, fg_dataset, bg_dataset):
        self.fg_dataset = fg_dataset
        self.bg_dataset = bg_dataset

    def __getitem__(self, index):
        img_fg, labels = self.fg_dataset[index]
        img_bg, _ = self.bg_dataset[index]
        return (img_fg, img_bg, labels)

    def __len__(self):
        return min(len(self.fg_dataset), len(self.bg_dataset))
    
    
class GrassyMNIST(VisionDataset):
    def __init__(self, root, split, w=0.5, cropping=True, loader=default_loader, transform=None, target_transform=None, download=True):
        super(GrassyMNIST, self).__init__(root, transform=transform, target_transform=target_transform)      
        self.w = w
        self.files = {'train': 'target_train_w={}.pt'.format(self.w), 
                      'test': 'target_test_w={}.pt'.format(self.w), 
                      'background': 'background_w={}.pt'.format(self.w)}
        
        self.cropping = cropping
        if self.cropping:
            self.bg_transform = transforms.RandomCrop((28,28))
            for k, v in self.files.items():
                self.files[k] = v[:-3] + '_cropped.pt'
        else:
            bg_transform = transforms.Resize((28,28))
            
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = torch.load(join(self.data_path, self.files[split]))
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img.numpy())
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
    
    @property
    def data_path(self):
        return join(self.root, self.__class__.__name__)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def download(self):
        if self._check_exists():
            return
        self.compose_sets()
        
    def _check_exists(self):
        for k, v in self.files.items():
            if not os.path.exists(join(self.data_path, v)):
                return False
        return True
    
    def compose_sets(self):
        """
        Load mnist and grass datasets, and then generate target dataset and background dataset
        """
        dataset_mnist_train = datasets.MNIST(self.root, train=True, transform=transforms.ToTensor(), download=True)
        dataset_mnist_test = datasets.MNIST(self.root, train=False, transform=transforms.ToTensor(), download=True)
        dataset_grass = Load_Grass(self.root, transform=transforms.ToTensor())

        bg_imgs = dataset_grass
        bg_split = torch.randperm(bg_imgs.shape[0])
        bg_imgs = dataset_grass[bg_split[(bg_imgs.shape[0]//2):]]
        target_bg_imgs = dataset_grass[bg_split[:(bg_imgs.shape[0]//2)]]
        output_dir = join(self.root, 'GrassyMNIST')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
#         torch.save(bg_imgs, join(output_dir, self.files['background']))
        self.compose_set(dataset_mnist_train, target_bg_imgs, join(output_dir, self.files['train']))
        self.compose_set(dataset_mnist_test, target_bg_imgs, join(output_dir, self.files['test']))
        self.compose_set(None, bg_imgs, join(output_dir, self.files['background']))
        
    def compose_set(self, dataset_mnist, target_bg_imgs, output_path):
        composed_imgs = []
        
        if dataset_mnist is None:
            N = 60000
            labels = torch.ones(N) * -1
        else:
            mnist_imgs = []
            labels = dataset_mnist.targets
            for img in tqdm(dataset_mnist.data, desc="Processing MNIST"):
                img = Image.fromarray(img.numpy(), mode='L')
                img = transforms.ToTensor()(img)
                mnist_imgs.append(img)
            mnist_imgs = torch.stack(mnist_imgs, 0)
            N = mnist_imgs.shape[0]
            
        M = target_bg_imgs.shape[0]
        bg_idx = []
        for i in range(N // M + 1):
            bg_idx.append(torch.randperm(M))
        bg_idx = torch.cat(bg_idx, 0)
                
        for n in tqdm(range(N), desc="Composing {}.".format(output_path.split('/')[-1][:-3])):
            bg_img = self.bg_transform(target_bg_imgs[bg_idx[n]])
            if dataset_mnist is None:
                composed_img = bg_img
            else:
                composed_img = self.w * mnist_imgs[n] + (1.0 - self.w) * bg_img
            composed_imgs.append(composed_img)
                
        composed_imgs = torch.cat(composed_imgs, 0)
        torch.save((composed_imgs, labels), output_path)

def Load_Grass(root, transform):
    """
    Load Grass dataset
    """
    data_path = join(root, 'Grass')
    raw_folder = join(data_path, 'raw')
    processed_folder = join(data_path, 'processed')
    processed_path = join(processed_folder, 'data.pt')
    if os.path.exists(processed_path):
        return torch.load(processed_path)
    else:
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        print('Loading Grass...')
        data = []
        for f in range(892):
            img = Image.open(join(processed_folder, 'jpg', '%04d.JPG' % (f+1)))
            img = img.convert(mode='L')
            data.append(transform(img))
        data = torch.stack(data, 0)
        assert data.shape == (892, 1, 100, 100)
        with open(processed_path, 'wb') as f:
            torch.save(data, f)
        return data
    
    
# class FlowerMNIST(VisionDataset):
#     def __init__(self, root, train, loader=default_loader, transform=None, target_transform=None, download=True):
#         super(FlowerMNIST, self).__init__(root, transform=transform, target_transform=target_transform)      
        
#         self.files = ['target_train.pt', 'target_test.pt', 'background.pt']
#         if download:
#             self.download()

#         if not self._check_exists():
#             raise RuntimeError('Dataset not found.' +
#                                ' You can use download=True to download it')
        
#         self.data, self.labels = torch.load(join(self.data_path, 'target_train.pt') 
#                                             if train else join(self.data_path, 'target_test.pt'))
        
#     def __getitem__(self, index):
#         img, label = self.data[index], int(self.labels[index])
#         if self.transform is not None:
#             img = self.transform(img.numpy())
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return img, label

#     def __len__(self):
#         return len(self.data)
    
#     @property
#     def data_path(self):
#         return join(self.root, self.__class__.__name__)

#     @property
#     def class_to_idx(self):
#         return {_class: i for i, _class in enumerate(self.classes)}

#     def download(self):
#         if self._check_exists():
#             return
#         compose_flower_mnist(self.root)
        
#     def _check_exists(self):
#         for filename in self.files:
#             if not os.path.exists(join(self.data_path, filename)):
#                 return False
#         return True
    
# def compose_flower_mnist(root):
#     """
#     Load mnist and flowers datasets, and then generate target dataset and background dataset
#     """
#     dataset_mnist_train = datasets.MNIST(root, train=True, transform=transforms.ToTensor(), download=True)
#     dataset_mnist_test = datasets.MNIST(root, train=False, transform=transforms.ToTensor(), download=True)
#     dataset_flowers102 = Load_Flowers102(root)

#     bg_imgs = dataset_flowers102
#     bg_split = torch.randperm(bg_imgs.shape[0])
#     bg_imgs = dataset_flowers102[bg_split[(bg_imgs.shape[0]//2):]]
#     target_bg_imgs = dataset_flowers102[bg_split[:(bg_imgs.shape[0]//2)]]
#     output_dir = join(root, 'FlowerMNIST')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     torch.save(bg_imgs, join(output_dir, 'background.pt'))

#     def compose_subset(dataset_mnist, target_bg_imgs, output_dir, prefix):
#         mnist_imgs = []
#         mnist_labels = dataset_mnist.targets
#         for img in tqdm(dataset_mnist.data, desc="Processing MNIST"):
#             img = Image.fromarray(img.numpy(), mode='L')
#             img = transforms.ToTensor()(img)
#             mnist_imgs.append(img)
#         mnist_imgs = torch.stack(mnist_imgs, 0)
#         N = mnist_imgs.shape[0]
#         M = target_bg_imgs.shape[0]
#         bg_idx = []
#         for i in range(N // M + 1):
#             bg_idx.append(torch.randperm(M))
#         bg_idx = torch.cat(bg_idx, 0)
#         composed_imgs = []
#         for n in tqdm(range(N), desc="Composing Images.."):
#             composed_img = (mnist_imgs[n]*0.5 + target_bg_imgs[bg_idx[n]]) / 2.0
#             composed_imgs.append(composed_img)
#         composed_imgs = torch.cat(composed_imgs, 0)
#         torch.save((composed_imgs, mnist_labels), join(output_dir, 'target_{}.pt'.format(prefix)))
        
#     compose_subset(dataset_mnist_train, target_bg_imgs, output_dir, 'train')
#     compose_subset(dataset_mnist_test, target_bg_imgs, output_dir, 'test')
        
        
# def Load_Flowers102(root):
#     """
#     Download and preprocess 102Flower if if doesn't exist
#     """
#     data_path = join(root, 'Flowers102')
#     raw_folder = join(data_path, 'raw')
#     processed_folder = join(data_path, 'processed')
#     processed_path = join(processed_folder, 'data.pt')
#     data_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'
#     if os.path.exists(processed_path):
#         return torch.load(processed_path)
#     else:
#         try:
#             from urllib.request import urlretrieve
#         except ImportError:
#             from urllib import urlretrieve
#         os.makedirs(raw_folder, exist_ok=True)
#         os.makedirs(processed_folder, exist_ok=True)
#         print('Downloading images from %s ...' % data_url)
#         image_file = join(raw_folder, "102flowers.tgz")
#         urlretrieve(data_url + "/102flowers.tgz", image_file)
#         tarfile.open(image_file).extractall(path=raw_folder)
#         os.remove(image_file)
#         print('Processing...')
#         pre_transforms = transforms.Compose([
#                             transforms.Resize((28,28)),
#                             transforms.ToTensor()])
#         data = []
#         for f in range(8189):
#             img = Image.open(join(raw_folder, 'jpg', 'image_0%04d.jpg' % (f+1)))
#             img = img.convert(mode='L')
#             data.append(pre_transforms(img))
#         data = torch.stack(data, 0)
#         assert data.shape == (8189, 1, 28, 28)
#         with open(processed_path, 'wb') as f:
#             torch.save(data, f)
#         return data