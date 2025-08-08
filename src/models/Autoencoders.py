import torch
from torch import nn

class AE(nn.Module):
    """Autoencoder: encodes & decodes, returning both reconstruction & bottleneck."""
    def __init__(self, in_ch: int, out_ch: int, features: list[int] = [16,32,64,128]) -> None:
        super().__init__()
        # build encoder
        self.encoder = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_ch, f, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(f),
                nn.LeakyReLU(),
            ) for f in features
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], 3, 1, 1),
            nn.BatchNorm2d(features[-1]),
            nn.LeakyReLU(),
        )
        # build decoder
        reversed_feats = list(reversed(features))
        decoder_layers = []
        prev = features[-1]
        for nxt in reversed_feats:
            decoder_layers += [
                nn.ConvTranspose2d(prev, nxt, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(nxt),
                nn.LeakyReLU(),
            ]
            prev = nxt
        decoder_layers.append(nn.ConvTranspose2d(prev, out_ch, 3, 2, 1, output_padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        recon = self.decoder(bottleneck)
        return recon, bottleneck
