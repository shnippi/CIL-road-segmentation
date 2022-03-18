import torchvision.transforms.functional as TF

def get_transforms(config):

    IMAGE_HEIGHT = config['image_height']
    IMAGE_WIDTH = config['image_width']

    train_transform = TF.Compose(
        [
            TF.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            TF.Rotate(limit=35, p=1.0),
            TF.HorizontalFlip(p=0.5),
            TF.VerticalFlip(p=0.1),
            TF.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            TF.ToTensor(),
        ],
    )

    val_transforms = TF.Compose(
        [
            TF.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            TF.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            TF.ToTensor(),
        ],
    )

    return train_transform, val_transforms
