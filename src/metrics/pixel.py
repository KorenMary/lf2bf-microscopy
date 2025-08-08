"""Pixel-level metrics for image evaluation."""
from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy float32 array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype.kind in ('u', 'i'):
        x = x.astype(np.float32)
    elif x.dtype.kind != 'f':
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
    return x


def _infer_channel_axis(x: np.ndarray) -> Optional[int]:
    """Infer channel axis for 2D/3D images. Returns None for 2D."""
    if x.ndim == 2:
        return None
    if x.ndim == 3:
        if x.shape[-1] <= 4:
            return -1
        if x.shape[0] <= 4:
            return 0
        return -1
    raise ValueError(f"Unsupported image ndim {x.ndim}; expected 2D/3D image.")


def _maybe_move_channels_first_to_last(gt: np.ndarray, pr: np.ndarray, ch_axis: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """For old skimage API that only supports channels-last via `multichannel=True`."""
    if ch_axis == 0:
        return np.moveaxis(gt, 0, -1), np.moveaxis(pr, 0, -1), True
    return gt, pr, (ch_axis == -1)


def _normalize_range_and_data(gt: np.ndarray, pred: np.ndarray, data_range: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (gt, pred, data_range) where data_range is numeric."""
    if data_range is None:
        v = max(float(np.max(gt)), float(np.max(pred)))
        dr = 1.0 if v <= 1.0 else 255.0
    else:
        dr = float(data_range)
    return gt, pred, dr


def psnr(gt: ArrayLike, pred: ArrayLike, data_range: Optional[float] = 1.0) -> float:
    gt_np = _to_numpy(gt)
    pr_np = _to_numpy(pred)
    gt_np, pr_np, dr = _normalize_range_and_data(gt_np, pr_np, data_range)
    return float(peak_signal_noise_ratio(gt_np, pr_np, data_range=dr))


def ssim(gt: ArrayLike, pred: ArrayLike, data_range: Optional[float] = 1.0, win_size: Optional[int] = None) -> float:
    """SSIM using skimage. Supports both new (channel_axis) and old (multichannel) APIs."""
    gt_np = _to_numpy(gt)
    pr_np = _to_numpy(pred)
    ch_axis = _infer_channel_axis(gt_np)
    gt_np, pr_np, dr = _normalize_range_and_data(gt_np, pr_np, data_range)
    try:
        return float(structural_similarity(gt_np, pr_np, data_range=dr, channel_axis=ch_axis, win_size=win_size))
    except TypeError:
        gt_np2, pr_np2, is_multich = _maybe_move_channels_first_to_last(gt_np, pr_np, ch_axis if ch_axis is not None else -1)
        return float(structural_similarity(gt_np2, pr_np2, data_range=dr, multichannel=is_multich, win_size=win_size))


def mse(gt: ArrayLike, pred: ArrayLike) -> float:
    gt_np = _to_numpy(gt)
    pr_np = _to_numpy(pred)
    return float(np.mean((gt_np - pr_np) ** 2))


def mae(gt: ArrayLike, pred: ArrayLike) -> float:
    gt_np = _to_numpy(gt)
    pr_np = _to_numpy(pred)
    return float(np.mean(np.abs(gt_np - pr_np)))


def total_variation(img: ArrayLike, isotropic: bool = True) -> float:
    """Total variation of a single image (2D) or a 3D image with channels."""
    x = _to_numpy(img)
    if x.ndim == 3:
        if _infer_channel_axis(x) == 0:
            chans = [x[c] for c in range(x.shape[0])]
        else:
            chans = [x[..., c] for c in range(x.shape[-1])]
        return float(sum(total_variation(c, isotropic=isotropic) for c in chans))
    if x.ndim != 2:
        raise ValueError("total_variation expects 2D image or 3D with channels.")
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)
    if isotropic:
        tv = np.sum(np.sqrt(dx[:, :-1] ** 2 + dy[:-1, :] ** 2))
    else:
        tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    return float(tv / x.size)


def l1_masked_areas(gt: ArrayLike, pred: ArrayLike, mask: ArrayLike) -> float:
    """Mean absolute error restricted to mask==1 (or >0) pixels."""
    gt_np = _to_numpy(gt)
    pr_np = _to_numpy(pred)
    m = _to_numpy(mask)
    m = (m > 0).astype(np.float32)
    denom = max(1.0, float(m.sum()))
    return float(np.sum(np.abs(gt_np - pr_np) * m) / denom)

