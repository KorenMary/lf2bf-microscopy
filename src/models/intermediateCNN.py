from torch import nn

class IntermediateCNN(nn.Module):
    """Maps one bottleneck tensor to another via 5×5 conv blocks."""
    def __init__(self, channels: int, depth: int = 5) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
