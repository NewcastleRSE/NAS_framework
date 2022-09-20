import torch
from torchvision import datasets
import os


class Image_Download:
    def __init__(self):
        pass

    def get_data(data_info, BATCHSIZE, train_transforms, test_transforms):
        DIR = os.getcwd()
        dataclass = eval(f"datasets.{data_info}")

        image_train = dataclass(
            DIR, train=True, download=True, transform=train_transforms
        )
        image_valid = dataclass(
            DIR, train=False, download=True, transform=test_transforms
        )
        image_test = dataclass(
            DIR, train=False, download=True, transform=test_transforms
        )

        n_classes = len(image_train.targets.unique())
        n_in = image_train.data[0].unsqueeze(0).shape[0]

        train_loader = torch.utils.data.DataLoader(
            image_train, batch_size=BATCHSIZE, num_workers=4
        )
        valid_loader = torch.utils.data.DataLoader(
            image_valid, batch_size=BATCHSIZE, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            image_test, batch_size=BATCHSIZE, num_workers=4
        )

        return train_loader, valid_loader, test_loader, n_classes, n_in
