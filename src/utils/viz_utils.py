import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio

from train_utils import natsort


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_gray_tensor(path: str) -> torch.Tensor:
    """Load image as [1,1,H,W] float32 in [0,1]. Supports grayscale or RGB (uses first channel)."""
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = torch.from_numpy(img).float()
    if img.max() > 1.0:
        img = img / 255.0
    return img.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_bf_from_lf(autoencoder_lf, autoencoder_bf, intermediate_cnn, lf: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run LF→latent→BF-latent→BF path and return (pred_bf, z_lf, z_bf)."""
    autoencoder_lf.eval(); autoencoder_bf.eval(); intermediate_cnn.eval()
    lf = lf.to(device)
    _, z_lf = autoencoder_lf(lf)
    z_bf = intermediate_cnn(z_lf)
    pred_bf = autoencoder_bf.decoder(z_bf).clamp(0, 1)
    return pred_bf, z_lf, z_bf


def save_triplet_figure(lf: torch.Tensor, bf: torch.Tensor, pred_bf: torch.Tensor, out_path: str,
                        title_fontsize: int = 24, figsize: Tuple[int,int] = (18, 6)):
    """Save a 3-panel figure: GT LF, GT BF, Prediction (+PSNR)."""
    lf_np = lf[0, 0].detach().cpu().numpy()
    bf_np = bf[0, 0].detach().cpu().numpy()
    pr_np = pred_bf[0, 0].detach().cpu().numpy()

    psnr_pred = peak_signal_noise_ratio(bf_np, pr_np, data_range=1.0)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(lf_np, cmap='gray'); ax[0].set_title('GT LF', fontsize=title_fontsize); ax[0].axis('off')
    ax[1].imshow(bf_np, cmap='gray'); ax[1].set_title('GT BF', fontsize=title_fontsize); ax[1].axis('off')
    ax[2].imshow(pr_np, cmap='gray'); ax[2].set_title(f'Prediction: {psnr_pred:.2f} dB', fontsize=title_fontsize); ax[2].axis('off')
    fig.tight_layout(); plt.savefig(out_path); plt.close(fig)


@torch.no_grad()
def save_frames(input_root: str, dest_dir: str, autoencoder_lf, autoencoder_bf, intermediate_cnn,
                device: Optional[str] = None, limit: Optional[int] = None, numeric_sort: bool = True):
    """Render side-by-side frames for test pairs and save them to disk.

    Expects `<root>/test_A` and `<root>/test_B` with matching filenames.
    """
    lf_dir = os.path.join(input_root, 'test_A')
    bf_dir = os.path.join(input_root, 'test_B')
    if not os.path.isdir(lf_dir) or not os.path.isdir(bf_dir):
        raise FileNotFoundError('Expected test_A and test_B under the dataset root.')

    ensure_dir(dest_dir)

    files = [f for f in os.listdir(lf_dir) if not f.startswith('.')]
    files = natsort(files) if numeric_sort else sorted(files)

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder_lf = autoencoder_lf.to(dev).eval()
    autoencoder_bf = autoencoder_bf.to(dev).eval()
    intermediate_cnn = intermediate_cnn.to(dev).eval()

    count = 0
    for file_name in files:
        lf_path = os.path.join(lf_dir, file_name)
        bf_path = os.path.join(bf_dir, file_name)
        if not os.path.exists(bf_path):
            continue

        lf = load_gray_tensor(lf_path)
        bf = load_gray_tensor(bf_path)

        pred_bf, _, _ = predict_bf_from_lf(autoencoder_lf, autoencoder_bf, intermediate_cnn, lf, dev)

        out_path = os.path.join(dest_dir, os.path.splitext(file_name)[0] + '.png')
        save_triplet_figure(lf, bf, pred_bf, out_path)

        count += 1
        if limit is not None and count >= limit:
            break

    print(f'Saved {count} frame(s) to {dest_dir}')


def images_to_video(input_folder: str, output_video: str, fps: int = 30, codec: str = 'mp4v'):
    """Stitch images (png/jpg) into a video. Images are sorted numerically by stem.
    Defaults to MP4 (mp4v). Use codec 'XVID' and .avi suffix if you prefer AVI.
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]
    if not image_files:
        raise FileNotFoundError('No images found in input folder.')

    image_files = natsort(image_files)
    first = cv2.imread(os.path.join(input_folder, image_files[0]))
    if first is None:
        raise RuntimeError('Failed to read first image.')

    height, width = first.shape[:2]

    if output_video.lower().endswith('.avi') and codec == 'mp4v':
        codec = 'XVID'

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for name in image_files:
        frame = cv2.imread(os.path.join(input_folder, name))
        if frame is None:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(frame)

    out.release()
    print(f'Video created: {output_video}')
