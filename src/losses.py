from typing import Callable
import torch
from torch import nn

try:
    from pytorch_msssim import SSIM as _SSIM
    _has_ssim = True
except Exception:
    _has_ssim = False


class SSIM(nn.Module):
    """Wrapper: uses pytorch-msssim SSIM if available, else falls back to MSE."""
    def __init__(self, data_range: float = 1.0, channel: int = 1):
        super().__init__()
        if _has_ssim:
            self.impl = _SSIM(data_range=data_range, channel=channel)
        else:
            self.impl = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.impl(x, y)


def reconstruction_loss() -> Callable:
    return nn.MSELoss()
