from torchvision import transforms

def get_transforms(config):

    IMAGE_HEIGHT = config['image_height']
    IMAGE_WIDTH = config['image_width']

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

    return train_transform, val_transforms
