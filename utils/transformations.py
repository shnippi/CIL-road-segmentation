"""
#
# This file contains transformation functions for normalizing and resizing of the data set
#  
"""

from torchvision import transforms

def get_transforms(config):

    IMAGE_HEIGHT = config['image_height']
    IMAGE_WIDTH = config['image_width']

    if config['transformation'] == 'rgb':
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                #transforms.RandomRotation(10),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ],
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ],
        )
    elif config['transformation'] == 'label':
        train_transform_A = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                #transforms.RandomRotation(10),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ],
        )

        train_transform_B = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                #transforms.RandomRotation(10),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
            ],
        )

        val_transforms_A = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ],
        )

        val_transforms_B = transforms.Compose(
            [
                transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
            ],
        )

        train_transform = [train_transform_A, train_transform_B]
        val_transforms = [val_transforms_A, val_transforms_B]

    return train_transform, val_transforms
