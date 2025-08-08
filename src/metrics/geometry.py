"""Geometry / instance-level metrics: centroids, Chamfer, Hausdorff."""
from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from skimage.measure import label, regionprops

try:
    from skimage.metrics import hausdorff_distance as _sk_hausdorff_distance
    _HAS_SK_HAUSDORFF = True
except Exception:
    _HAS_SK_HAUSDORFF = False

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
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


def _to_binary_mask(x: ArrayLike) -> np.ndarray:
    m = _to_numpy(x)
    if m.ndim != 2:
        raise ValueError("Binary mask must be 2D (H, W).")
    if m.dtype.kind == 'f':
        m = m > 0.5
    else:
        m = m > 0
    return m.astype(bool)


def extract_cell_centers(mask: ArrayLike, min_area: int = 5) -> np.ndarray:
    """Extract centroids (y, x) from a binary or labeled mask using skimage.regionprops."""
    m = _to_numpy(mask)
    if m.ndim != 2:
        raise ValueError("extract_cell_centers expects a 2D mask.")
    if m.dtype.kind == 'f':
        m = m > 0.5
    else:
        m = m > 0
    lab = label(m.astype(bool))
    centers: List[Tuple[float, float]] = []
    for r in regionprops(lab):
        if r.area >= min_area:
            cy, cx = r.centroid
            centers.append((float(cy), float(cx)))
    if not centers:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def chamfer_distance_points(a: ArrayLike, b: ArrayLike, reduce: str = "mean") -> float:
    """Symmetric Chamfer distance between two point sets a[N,2], b[M,2] in (y,x)."""
    A = _to_numpy(a).reshape(-1, 2)
    B = _to_numpy(b).reshape(-1, 2)
    if len(A) == 0 and len(B) == 0:
        return 0.0
    if len(A) == 0 or len(B) == 0:
        return float('inf')
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    d_ab = np.min(d2, axis=1)
    d_ba = np.min(d2, axis=0)
    if reduce == "mean":
        return float(np.mean(d_ab) + np.mean(d_ba))
    elif reduce == "max":
        return float(np.max(d_ab) + np.max(d_ba))
    else:
        raise ValueError("reduce must be 'mean' or 'max'")


def chamfer_distance_masks(a_mask: ArrayLike, b_mask: ArrayLike, reduce: str = "mean") -> float:
    """Chamfer distance between two binary masks using Euclidean Distance Transforms.
    Requires SciPy for EDT; falls back to point-based Chamfer otherwise.
    """
    A = _to_binary_mask(a_mask)
    B = _to_binary_mask(b_mask)
    if _HAS_SCIPY:
        dist_to_B = distance_transform_edt(~B)
        dist_to_A = distance_transform_edt(~A)
        Ay, Ax = np.nonzero(A)
        By, Bx = np.nonzero(B)
        if Ay.size == 0 and By.size == 0:
            return 0.0
        if Ay.size == 0 or By.size == 0:
            return float('inf')
        d_ab = dist_to_B[Ay, Ax] ** 2
        d_ba = dist_to_A[By, Bx] ** 2
        if reduce == "mean":
            return float(d_ab.mean() + d_ba.mean())
        elif reduce == "max":
            return float(d_ab.max() + d_ba.max())
        else:
            raise ValueError("reduce must be 'mean' or 'max'")
    else:
        a_pts = np.column_stack(np.nonzero(A)).astype(np.float32)
        b_pts = np.column_stack(np.nonzero(B)).astype(np.float32)
        return chamfer_distance_points(a_pts, b_pts, reduce=reduce)


def chamfer_distance_cell_centers(a_mask: ArrayLike, b_mask: ArrayLike, min_area: int = 5, reduce: str = "mean") -> float:
    """Chamfer distance between cell-centers extracted from two masks."""
    A = extract_cell_centers(a_mask, min_area=min_area)
    B = extract_cell_centers(b_mask, min_area=min_area)
    return chamfer_distance_points(A, B, reduce=reduce)


def hausdorff_distance_masks(a_mask: ArrayLike, b_mask: ArrayLike) -> float:
    """Hausdorff distance (symmetric) between two binary masks."""
    A = _to_binary_mask(a_mask)
    B = _to_binary_mask(b_mask)
    if A.size == 0 and B.size == 0:
        return 0.0
    if _HAS_SK_HAUSDORFF:
        return float(_sk_hausdorff_distance(A, B))
    # Fallback: naive over foreground coordinates
    Ayx = np.column_stack(np.nonzero(A)).astype(np.float32)
    Byx = np.column_stack(np.nonzero(B)).astype(np.float32)
    if Ayx.size == 0 or Byx.size == 0:
        return float('inf')
    d2 = ((Ayx[:, None, :] - Byx[None, :, :]) ** 2).sum(axis=2)
    d_ab = np.sqrt(np.min(d2, axis=1)).max()
    d_ba = np.sqrt(np.min(d2, axis=0)).max()
    return float(max(d_ab, d_ba))



