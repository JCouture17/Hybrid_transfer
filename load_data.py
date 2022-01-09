### Load Data ###
from misc_functions import functions
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

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
    
    def load_datasets(batch_size):
        fcts = functions() 
        training_data = torch.tensor(fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_his.mat',
                                  'training_his').astype(np.float16))
        testing_data = torch.tensor(fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_his.mat',
                                 'testing_his').astype(np.float16))
        training_targets = torch.tensor(fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_targets.mat',
                                     'training_targets').astype(np.int16))
        testing_targets = torch.tensor(fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_targets.mat',
                                    'testing_targets').astype(np.int16))
        
        training_data = training_data.reshape([training_data.shape[0], 1, training_data.shape[1]])
        testing_data = testing_data.reshape([testing_data.shape[0], 1, testing_data.shape[1]])
        
        train_loader = CustomDataset(dataset=(training_data, training_targets))
        test_loader = CustomDataset(dataset=(testing_data, testing_targets))
        train_loader = DataLoader(train_loader, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_loader, shuffle=True, batch_size=batch_size)
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
    

class CustomDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, index):
        x = self.dataset[0][index]
        y = self.dataset[1][index]
        return x, y