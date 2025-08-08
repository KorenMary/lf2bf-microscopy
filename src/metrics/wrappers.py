"""Convenience wrappers for computing metrics over single pairs or batches."""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from .pixel import psnr, ssim, mse, mae, total_variation, l1_masked_areas

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def calculate_metrics(
    gt: ArrayLike,
    pred: ArrayLike,
    *,
    mask: Optional[ArrayLike] = None,
    data_range: Optional[float] = 1.0,
    with_tv: bool = False,
) -> Dict[str, float]:
    out = {
        'psnr': psnr(gt, pred, data_range=data_range),
        'ssim': ssim(gt, pred, data_range=data_range),
        'mse': mse(gt, pred),
    }
    if mask is not None:
        out['mae_masked'] = l1_masked_areas(gt, pred, mask)
    else:
        out['mae'] = mae(gt, pred)
    if with_tv:
        out['tv_pred'] = total_variation(pred)
    return out


def batch_metrics(
    gt_batch: ArrayLike,
    pred_batch: ArrayLike,
    *,
    mask_batch: Optional[ArrayLike] = None,
    data_range: Optional[float] = 1.0,
    reduce: str = 'mean',
    with_tv: bool = False,
) -> Dict[str, float]:
    GT = _to_numpy(gt_batch)
    PR = _to_numpy(pred_batch)
    if GT.shape != PR.shape:
        raise ValueError(f"Shapes must match, got {GT.shape} vs {PR.shape}")
    if GT.ndim < 3:
        raise ValueError("Expected a batch: at least [N,H,W].")
    N = GT.shape[0]

    vals: Dict[str, List[float]] = {}
    for i in range(N):
        m = None if mask_batch is None else _to_numpy(mask_batch)[i]
        res = calculate_metrics(GT[i], PR[i], mask=m, data_range=data_range, with_tv=with_tv)
        for k, v in res.items():
            vals.setdefault(k, []).append(float(v))

    if reduce == 'mean':
        return {k: float(np.mean(v)) for k, v in vals.items()}
    elif reduce == 'median':
        return {k: float(np.median(v)) for k, v in vals.items()}
    else:
        raise ValueError("reduce must be 'mean' or 'median'")


__all__ = ['calculate_metrics', 'batch_metrics']
