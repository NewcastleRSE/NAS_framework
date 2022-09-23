from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data, label, transform):
        super(MyDataset, self).__init__()

        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = np.moveaxis(x, 0, 2)
        x = x.astype(np.uint8)
        x = Image.fromarray(x)

        y = self.label[index]

        return self.transform(x), y


class Image_Data:
    def __init__(self):
        pass

    def get_data(data_info, BATCHSIZE, train_transforms, test_transforms):
        print(data_info)
        train_x = np.load(os.path.join(data_info, "train_x.npy"))
        train_y = np.load(os.path.join(data_info, "train_y.npy"))
        valid_x = np.load(os.path.join(data_info, "valid_x.npy"))
        valid_y = np.load(os.path.join(data_info, "valid_y.npy"))
        test_x = np.load(os.path.join(data_info, "test_x.npy"))
        test_y = np.load(os.path.join(data_info, "test_y.npy"))

        with open(os.path.join(data_info, "dataset_metadata"), "r") as f:
            metadata = json.load(f)

        n_classes = metadata["n_classes"]
        n_in = train_x.shape[1]

        train_dataset = MyDataset(train_x, train_y, train_transforms)
        valid_dataset = MyDataset(valid_x, valid_y, test_transforms)
        test_dataset = MyDataset(test_x, test_y, test_transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCHSIZE, num_workers=4
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=BATCHSIZE, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCHSIZE, num_workers=4
        )

        return train_loader, valid_loader, test_loader, n_classes, n_in
