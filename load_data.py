### Load Data ###
from misc_functions import functions
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn

class data:
    def load_images(train_dataset, train_targets, test_dataset, test_targets, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        # Build dataset
        train_data = CustomImageDataset(dataset=(train_dataset, train_targets), transform=transform)
        test_data = CustomImageDataset(dataset=(test_dataset, test_targets), transform=transform)
        # Build dataloader
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        return train_loader, test_loader
    
    def load_dataset(batch_size):
        from skimage import io as skio
        ld = functions()
        
        train_data = skio.imread('./Matlab/train_test_images\\training_data.tif')
        test_data = skio.imread('./Matlab/train_test_images\\testing_data.tif')

        train_rul = ld.load('./Matlab/train_test_images\\training_targets.mat', 'rul').astype(np.int16)
        test_rul = ld.load('./Matlab/train_test_images\\testing_targets.mat', 'rul').astype(np.int16)

        train_loader, test_loader = data.load_images(train_data, train_rul, test_data, test_rul, batch_size)
        
        return train_loader, test_loader
    
class CustomImageDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, index):
        x = self.dataset[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.dataset[1][index]
        return x, y
    
    