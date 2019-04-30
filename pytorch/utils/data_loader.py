import os
import torch
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

class DatasetA(Dataset):
    def __init__(self, params, train=True):
        self.train = train
        t = transforms.Compose([ transforms.Resize(params.image_size),
                                  transforms.ToTensor() ])
        if self.train:
            self.data = ImageFolder(root=params.train_path_a, transform=t, train=True)
        else:
            self.data = ImageFolder(root=params.test_path_a, transform=t, train=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample is already in range [0,1] want in range [-1,1]
        if self.train:
            sample = self.data[idx]
            sample = (2 * sample) - 1.0
            return sample
        else:
            path, sample = self.data[idx]
            sample = (2 * sample) - 1.0
            return path, sample

class DatasetB(Dataset):
    def __init__(self, params, train=True):
        self.train = train
        t = transforms.Compose([ transforms.Resize(params.image_size),
                                  transforms.ToTensor() ])
        if self.train:
            self.data = ImageFolder(root=params.train_path_b, transform=t, train=True)
        else:
            self.data = ImageFolder(root=params.test_path_b, transform=t, train=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample is already in range [0,1] want in range [-1,1]
        if self.train:
            sample = self.data[idx]
            sample = (2 * sample) - 1.0
            return sample
        else:
            path, sample = self.data[idx]
            sample = (2 * sample) - 1.0
            return path, sample

def get_datasets(params, train=True):
    return DatasetA(params, train), DatasetB(params, train)

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
