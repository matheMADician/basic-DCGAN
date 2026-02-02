from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torchvision.datasets import ImageFolder

dataset_classes = {
    'fashion-mnist': datasets.FashionMNIST,
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'fashionmnist': datasets.FashionMNIST,
    'svhn': datasets.SVHN,
    'stl10': datasets.STL10,
}

def Get_loader(
        data_class,
        batch_size,
        root= 'dataset/',
        train= True,
        download= True,
        shuffle= None,
        transform= None,
        do_flip= False,
        flip_prob= 0.5
    ):
    
    if(data_class not in dataset_classes and data_class != 'custom'): 
        print(f"data_class {data_class} not in dictionary")
        return
    
    if shuffle is None: shuffle= train

    test_transform, train_transform= transforms.ToTensor(), transforms.ToTensor()
    if transform is not None:
        test_transform = transform
        train_transform = transform
    elif do_flip:
        train_transform= transforms.Compose([
            transforms.RandomHorizontalFlip(p= flip_prob),
            transforms.ToTensor()
        ])
    else:
        train_transform = test_transform
    
    if data_class == 'custom':
        # For custom dataset from folder
        if not os.path.exists(root):
            print(f"Custom dataset path {root} does not exist. Please download it first.")
            return None
        Dataset = ImageFolder(root=root, transform=train_transform if train else test_transform)
        return DataLoader(
            dataset= Dataset,
            batch_size= batch_size,
            shuffle= shuffle
        )
    
    if data_class.lower() == 'svhn':
        split= 'train' if train else 'test'
        Dataset = dataset_classes[data_class.lower()](
            root= root,
            split= split,
            transform= train_transform if train else test_transform,
            download= download
        )
        return DataLoader(
            dataset= Dataset,
            batch_size= batch_size,
            shuffle= shuffle
        )
    
    Dataset = dataset_classes[data_class.lower()](
        root= root,
        train= train,
        transform= train_transform if train else test_transform,
        download= download)
    return DataLoader(
        dataset= Dataset,
        batch_size= batch_size,
        shuffle= shuffle
    )