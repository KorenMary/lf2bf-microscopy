import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage import io

class PHIOData(Dataset):
    """
    Paired lens-free (LF) / brightfield (BF) dataset for both AEs and CNN.
    Expects folder structure:
      <root_dir>/{train,val,test}_A/   ← LF images
      <root_dir>/{train,val,test}_B/   ← BF images
    """
    def __init__(self, root_dir: str, transforms=None, mode: str = "train"):
        super().__init__()
        self.root_dir = root_dir
        self.transforms_lf = transforms
        self.transforms_bf = transforms

        self.lensless_dir = os.path.join(root_dir, f"{mode}_A")
        self.brightfield_dir = os.path.join(root_dir, f"{mode}_B")

        self.file_names = sorted(os.listdir(self.lensless_dir))
        self.lf_paths = [os.path.join(self.lensless_dir, fn) for fn in self.file_names]
        self.bf_paths = [os.path.join(self.brightfield_dir, fn) for fn in self.file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int):
        lf = io.imread(self.lf_paths[idx])
        bf = io.imread(self.bf_paths[idx])
        if self.transforms_lf is not None:
            lf = self.transforms_lf(lf)
        if self.transforms_bf is not None:
            bf = self.transforms_bf(bf)
        # normalize from [0,255] → [0,1]
        return lf.float() / 255.0, bf.float() / 255.0
