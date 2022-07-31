"""
#
# This file contains functions from the original pix2pix to load their images
#  
"""

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class OriginalPix2Pix(Dataset):
    def __init__(self, root_dir, phase, transform):
        """
        Args:
            root_dir (string): Directory of the training data
            phase (sting): Either "train" or "val"
        """
        self.transform = transform

        root_dir = os.path.join(root_dir, phase)
        self.paths = sorted(create_path_list(root_dir))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = np.array(Image.open(img_path))

        image_A = Image.fromarray(image[:, :600, :])
        image_B = Image.fromarray(image[:, 600:, :])

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}


def create_path_list(dir):
    """ We make a list of all the paths in the given folder """

    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            paths.append(path)

    return paths
