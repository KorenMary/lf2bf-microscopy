import os
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage import io

from train_utils import natsort


class PHIOPairDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        transform_lf: Optional[Callable] = None,
        transform_bf: Optional[Callable] = None,
        ensure_grayscale: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform_lf = transform_lf or Compose([ToTensor()])
        self.transform_bf = transform_bf or self.transform_lf
        self.ensure_grayscale = ensure_grayscale

        self.lf_dir = os.path.join(root_dir, f"{mode}_A")
        self.bf_dir = os.path.join(root_dir, f"{mode}_B")

        if not os.path.isdir(self.lf_dir):
            raise FileNotFoundError(f"Missing directory: {self.lf_dir}")
        if not os.path.isdir(self.bf_dir):
            raise FileNotFoundError(f"Missing directory: {self.bf_dir}")

        names = natsort([f for f in os.listdir(self.lf_dir) if not f.startswith('.')])
        self.file_names = [n for n in names if os.path.exists(os.path.join(self.bf_dir, n))]
        if len(self.file_names) != len(names):
            missing = set(names) - set(self.file_names)
            print(f"[warn] Missing BF counterparts for {len(missing)} file(s). Example: {list(missing)[:5]}")

        self.lf_paths = [os.path.join(self.lf_dir, n) for n in self.file_names]
        self.bf_paths = [os.path.join(self.bf_dir, n) for n in self.file_names]

    def __len__(self) -> int:
        return len(self.file_names)

    def _load_gray(self, path: str) -> np.ndarray:
        img = io.imread(path)
        if self.ensure_grayscale and img.ndim == 3:
            img = img[..., 0]
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lf_img = self._load_gray(self.lf_paths[idx])
        bf_img = self._load_gray(self.bf_paths[idx])

        lf_img = self.transform_lf(lf_img)  # float32 [0,1], shape [1,H,W]
        bf_img = self.transform_bf(bf_img)
        return lf_img, bf_img
