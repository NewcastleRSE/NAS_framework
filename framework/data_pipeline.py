from torchvision import transforms
from torchvision.transforms.transforms import Compose
from numpy_image import Image_Data


class Data_Pipeline:
    def __init__(self, data_info, augment_style, BATCHSIZE):

        self.data_info = data_info
        self.augment_style = augment_style
        self.batchsize = BATCHSIZE

        (
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.n_classes,
            self.n_in,
        ) = self.data_process()

    def data_augmentation(self, augment_style: str):

        crop_size = 224

        # Setup data transforms:
        flip = transforms.RandomHorizontalFlip()
        resize = transforms.Resize((crop_size, crop_size))
        tensor_transform = transforms.ToTensor()
        # rotate = transforms.RandomRotation(degrees=10, resample=Image.BILINEAR)
        # crop = transforms.RandomCrop(crop_size)
        # #colorjitter = trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.1)
        # # colorjitter = transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5)
        # # colorjitter.probability = 0.9
        # shear = transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=20) # 20 degrees random shear
        # # greyscale = transforms.RandomGrayscale(p=1.0) # make everything greyscale

        if augment_style == "default":
            train_transforms = transforms.Compose([resize, tensor_transform])

        elif augment_style == "flip":
            train_transforms = transforms.Compose([resize, flip, tensor_transform])

        test_transforms = transforms.Compose(
            [transforms.Resize((crop_size, crop_size)), transforms.ToTensor()]
        )
        return train_transforms, test_transforms

    def data_process(self):

        train_transforms, test_transforms = self.data_augmentation(self.augment_style)

        (
            train_loader,
            valid_loader,
            test_loader,
            n_classes,
            n_in,
        ) = Image_Data.get_data(
            self.data_info, self.batchsize, train_transforms, test_transforms
        )

        return train_loader, valid_loader, test_loader, n_classes, n_in
