import os
from torch.utils.data import Dataset

class PairedDataset(Dataset):

    def __init__(self, root_A, root_B, phase):
        """
        Args:
            root_dir (string): Directory of the training data (subject already included)
            phase (sting): Either "train" or "val"
        """

        self.root_A = root_A
        self.root_B = root_B

        # List of all the image paths from A
        dir_A = os.path.join(root_A, phase)
        self.paths_A = sorted(create_path_list(dir_A))

        # List of all the image paths from B
        dir_B = os.path.join(root_B, phase)
        self.paths_B = sorted(create_path_list(dir_B))


    def __getitem__(self, idx):

        item_A = self.paths_A[idx]
        item_B = self.paths_B[idx]

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.paths_A) + len(self.paths_B)


def create_path_list(dir):
    """ We make a list of all the paths in the given folder """

    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            paths.append(path)

    return paths
