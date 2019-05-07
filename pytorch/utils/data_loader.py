import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image

class ImageFolder():
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.train = train
        self.transform = transform
        files = os.listdir(self.root)
        self.samples = [os.path.join(self.root, f) for f in files]
        self.samples.sort()

    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.train:
            return sample
        return path, sample

class MaskFolder():
    def __init__(self, root, size=None):
        self.root = root
        self.size = size
        files = os.listdir(self.root)
        self.samples = [os.path.join(self.root, f) for f in files]
        self.samples.sort()

    def loader(self, path):
        return np.load(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.size:
            sample = Image.fromarray(sample, mode='L')
            sample = np.array(sample.resize((self.size,self.size)))
        sample = sample[None,:,:]
        return torch.from_numpy(sample.astype(np.float32))

class DatasetA(Dataset):
    def __init__(self, params, train=True, mask=False):
        self.train = train
        self.mask = mask
        self.size = params.image_size
        t = transforms.Compose([ transforms.Resize(params.image_size),
                                  transforms.ToTensor() ])
        if self.train:
            self.data = ImageFolder(root=params.train_path_a, transform=t, train=True)
        else:
            self.data = ImageFolder(root=params.test_path_a, transform=t, train=False)

        self.masks = None
        if self.mask:
            self.masks = MaskFolder(root=params.train_mask_a, size=params.image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample is already in range [0,1] want in range [-1,1]
        if self.train:
            sample = self.data[idx]
            sample = (2 * sample) - 1.0
            if self.mask:
                return [sample, self.masks[idx]]
            else:
                return [sample, torch.ones(1,self.size,self.size)]
        else:
            path, sample = self.data[idx]
            sample = (2 * sample) - 1.0
            if self.mask:
                return [path, sample, self.masks[idx]]
            else:
                return [path, sample, torch.ones(1,self.size,self.size)]

class DatasetB(Dataset):
    def __init__(self, params, train=True, mask=False):
        self.train = train
        self.mask = mask
        self.size = params.image_size
        t = transforms.Compose([ transforms.Resize(params.image_size),
                                  transforms.ToTensor() ])
        if self.train:
            self.data = ImageFolder(root=params.train_path_b, transform=t, train=True)
        else:
            self.data = ImageFolder(root=params.test_path_b, transform=t, train=False)

        self.masks = None
        if self.mask:
            self.masks = MaskFolder(root=params.train_mask_b, size=params.image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample is already in range [0,1] want in range [-1,1]
        if self.train:
            sample = self.data[idx]
            sample = (2 * sample) - 1.0
            if self.mask:
                return [sample, self.masks[idx]]
            else:
                return [sample, torch.ones(1,self.size,self.size)]
        else:
            path, sample = self.data[idx]
            sample = (2 * sample) - 1.0
            if self.mask:
                return [path, sample, self.masks[idx]]
            else:
                return [path, sample, torch.ones(1,self.size,self.size)]

def get_datasets(params, train=True):
    return DatasetA(params, train, params.mask), DatasetB(params, train, params.mask)

if __name__=="__main__":
    from argparse import Namespace
    params = {'image_size':256, 'train_path_a' : 'datasets/monet2photo/trainA', 
              'train_path_b' : 'datasets/monet2photo/trainB',
              'test_path_a' : 'datasets/monet2photo/testA',
              'test_path_b' : 'datasets/monet2photo/testB'}
    params = Namespace(**params)
    print('Testing training dataset...')
    x_train = DatasetA(params, train=True)
    print(x_train[100])

    print('Testing test dataset...')
    x_test = DatasetA(params, train=False)
    print(x_test[100])

    print('Testing in a dataloader...')
    data = torch.utils.data.DataLoader(x_test, batch_size=4, shuffle=True)
    for i, d in enumerate(data):
        filenames = d[0]
        xs = d[1]
        print(filenames)
        print(torch.max(xs))
        print(torch.min(xs))
        break
