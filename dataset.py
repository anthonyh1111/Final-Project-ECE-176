import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config


def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataloaders():
    train_transform, test_transform = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.DEVICE.type == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.DEVICE.type == "cuda" else False
    )

    return train_loader, test_loader
