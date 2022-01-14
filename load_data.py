### Load Data ###
from misc_functions import functions
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from typing import Iterator, Optional,  TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)

class Sampler(Generic[T_co]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class RandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool
    
    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            # seed = int(torch.empty((), dtype=torch.int64).random_().item())
            seed = 10
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples
    

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
        train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        test_loader = DataLoader(test_data, sampler=RandomSampler(test_data), batch_size=batch_size)
        return train_loader, test_loader
    
    def load_data(batch_size):
        from skimage import io
        ld = functions()
        train_data = io.imread('./Data/training_data.tif')
        test_data = io.imread('./Data/testing_dataset.tif')
        train_rul = ld.load('./Data/training_targets.mat', 'training_targets').astype(np.int16)
        test_rul = ld.load('./Data/testing_targets.mat', 'testing_targets').astype(np.int16)
        train_loader, test_loader = data.load_images(train_data, train_rul, test_data, test_rul, batch_size)
        return train_loader, test_loader
    
    def load_datasets(batch_size):
        fcts = functions() 
        training_data = fcts.load('./Data/training_his.mat', 'training_his').astype(np.float16)
        testing_data = fcts.load('./Data/testing_his.mat', 'testing_his').astype(np.float16)
        training_targets = fcts.load('./Data/training_targets.mat', 'training_targets').astype(np.int16)
        training_data = torch.tensor(np.delete(training_data, 0, 1))
        testing_targets = fcts.load('./Data/testing_targets.mat', 'testing_targets').astype(np.int16)
        testing_data = torch.tensor(np.delete(testing_data, 0, 1))
        
        # training_data = training_data.reshape([training_data.shape[0], 1, training_data.shape[1]])
        # testing_data = testing_data.reshape([testing_data.shape[0], 1, testing_data.shape[1]])
        # training_targets = training_targets.reshape([training_targets.shape[0], 1])
        # testing_targets = testing_targets.reshape([testing_targets.shape[0], 1])
        
        train_loader = CustomDataset(dataset=(training_data, training_targets))
        test_loader = CustomDataset(dataset=(testing_data, testing_targets))
        train_loader = DataLoader(train_loader, sampler=RandomSampler(train_loader), batch_size=batch_size)
        test_loader = DataLoader(test_loader, sampler=RandomSampler(test_loader), batch_size=batch_size)
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