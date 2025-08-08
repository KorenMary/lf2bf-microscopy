import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from train_utils import set_seed
from data import PHIOPairDataset
from models.Autoencoders import AE
from models.intermediateCNN import IntermediateCNN
from losses import SSIM, reconstruction_loss


def train_one_epoch(
    loader,
    model_lf,
    model_bf,
    inter_model,
    opt_lf,
    opt_bf,
    opt_inter,
    loss_lf_fn,
    loss_bf_fn,
    loss_inter_fn,
    device,
    detach_lf: bool,
    log_every: int = 100,
):
    model_lf.train(); model_bf.train(); inter_model.train()
    pbar = tqdm(loader, desc="train", ncols=100)
    for step, (lf_img, bf_img) in enumerate(pbar, 1):
        lf_img = lf_img.to(device, non_blocking=True)
        bf_img = bf_img.to(device, non_blocking=True)

        opt_lf.zero_grad(set_to_none=True)
        opt_bf.zero_grad(set_to_none=True)
        opt_inter.zero_grad(set_to_none=True)

        preds_lf, z_lf = model_lf(lf_img)
        preds_bf, z_bf = model_bf(bf_img)

        loss_lf = loss_lf_fn(preds_lf, lf_img)
        loss_bf = loss_bf_fn(preds_bf, bf_img)

        z_map_in = z_lf.detach() if detach_lf else z_lf
        z_map = inter_model(z_map_in)
        loss_inter = loss_inter_fn(z_map, z_bf.detach())

        total = loss_lf + loss_bf + loss_inter
        total.backward()

        opt_lf.step(); opt_bf.step(); opt_inter.step()

        if step % log_every == 0:
            pbar.set_postfix({
                "lf": f"{loss_lf.item():.4f}",
                "bf": f"{loss_bf.item():.4f}",
                "inter": f"{loss_inter.item():.4f}",
                "avg": f"{(total.item()/3):.4f}",
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Dataset root with *_A/*_B folders')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--latent-ch', type=int, default=128)
    ap.add_argument('--detach-lf', action='store_true', help='Stop inter loss from updating LF AE')
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tfm = Compose([ToTensor()])
    dataset = PHIOPairDataset(args.root, mode='train', transform_lf=tfm, transform_bf=tfm)

    gen = torch.Generator().manual_seed(2023)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size], generator=gen)

    dl_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)

    # Models
    ae_lf = AE(in_channels=1, out_channels=1, latent_channels=args.latent_ch).to(device)
    ae_bf = AE(in_channels=1, out_channels=1, latent_channels=args.latent_ch).to(device)
    inter = IntermediateCNN(in_channels=args.latent_ch).to(device)

    # Optims & losses
    opt_lf = optim.Adam(ae_lf.parameters(), lr=args.lr)
    opt_bf = optim.Adam(ae_bf.parameters(), lr=args.lr)
    opt_inter = optim.Adam(inter.parameters(), lr=args.lr)

    loss_lf_fn = reconstruction_loss()
    loss_bf_fn = reconstruction_loss()
    loss_inter_fn = SSIM()  # falls back to MSE if pytorch-msssim missing

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_one_epoch(
            dl_train,
            ae_lf, ae_bf, inter,
            opt_lf, opt_bf, opt_inter,
            loss_lf_fn, loss_bf_fn, loss_inter_fn,
            device=device,
            detach_lf=args.detach_lf,
            log_every=100,
        )

    torch.save({"weights": inter.state_dict()}, "checkpoints/intermediate_cnn_ssim.chpt")


if __name__ == '__main__':
    main()
