import torch
from torchvision import datasets
import os


class Image_Folder:
    def __init__(self):
        pass

    def get_data(data_info, BATCHSIZE, train_transforms, test_transforms):
        print(data_info)
        image_train = datasets.ImageFolder(
            data_info + "/Train", transform=train_transforms
        )
        image_valid = datasets.ImageFolder(
            data_info + "/Valid", transform=test_transforms
        )
        image_test = datasets.ImageFolder(
            data_info + "/Test", transform=test_transforms
        )

        n_classes = len(set(image_train.targets))
        n_in = image_train[0][0].shape[0]

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
