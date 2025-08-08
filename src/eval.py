import os
import argparse

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from skimage.metrics import peak_signal_noise_ratio

from data import PHIOPairDataset
from models import AE, IntermediateCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--ckpt', default='checkpoints/intermediate_cnn_ssim.chpt')
    ap.add_argument('--latent-ch', type=int, default=128)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tfm = Compose([ToTensor()])
    testset = PHIOPairDataset(args.root, mode='test', transform_lf=tfm, transform_bf=tfm)
    dl_test = DataLoader(testset, batch_size=1, shuffle=False)

    ae_lf = AE(in_channels=1, out_channels=1, latent_channels=args.latent_ch).to(device).eval()
    ae_bf = AE(in_channels=1, out_channels=1, latent_channels=args.latent_ch).to(device).eval()
    inter = IntermediateCNN(in_channels=args.latent_ch).to(device).eval()

    if os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        inter.load_state_dict(state['weights'])
    else:
        print(f"[warn] ckpt not found: {args.ckpt}")

    lf, bf = next(iter(dl_test))
    lf = lf.to(device); bf = bf.to(device)

    with torch.no_grad():
        pred_lf, z_lf = ae_lf(lf)
        z_bf = inter(z_lf)
        pred_bf = ae_bf.decoder(z_bf).clamp(0,1)

    psnr_pred = peak_signal_noise_ratio(bf[0,0].cpu().numpy(), pred_bf[0,0].cpu().numpy(), data_range=1.0)

    os.makedirs('outputs', exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(bf[0,0].cpu(), cmap='gray'); ax[0].set_title('GT BF'); ax[0].axis('off')
    ax[1].imshow(pred_bf[0,0].cpu(), cmap='gray'); ax[1].set_title(f'Pred BF • PSNR={psnr_pred:.2f}dB'); ax[1].axis('off')
    fig.tight_layout()
    fig.savefig('outputs/example.png', dpi=150)
    print('Saved outputs/example.png')


if __name__ == '__main__':
    main()
