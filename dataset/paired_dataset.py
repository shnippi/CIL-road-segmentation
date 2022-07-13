import os
from torch.utils.data import Dataset
from PIL import Image

class PairedDataset(Dataset):

    def __init__(self, root_A, root_B, phase, transform):
        """
        Args:
            root_dir (string): Directory of the training data
            phase (sting): Either "train" or "val"
        """
        self.transform = transform

        # List of all the image paths from A
        dir_A = os.path.join(root_A, phase)
        self.paths_A = sorted(create_path_list(dir_A))

        # List of all the image paths from B
        dir_B = os.path.join(root_B, phase)
        self.paths_B = sorted(create_path_list(dir_B))

    def __getitem__(self, idx):
        item_A = self.paths_A[idx]
        item_B = self.paths_B[idx]

        #image_A = Image.open(item_A).convert('RGB')
        #image_B = Image.open(item_B).convert('RGB')
        image_A = Image.open(item_A)
        image_B = Image.open(item_B)

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}

    def __len__(self):
        '''
        Return the range of the idx. Note that we have N pictures from A and N pictures from B
        However in each iteration we use a picture from A and from B. Hence we only have N samples, and not 2N.
        '''
        return len(self.paths_A)


def create_path_list(dir):
    """ We make a list of all the paths in the given folder """

    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            paths.append(path)

    return paths
