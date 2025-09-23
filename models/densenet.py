import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= DENSE BLOCK =========================
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, 3, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )
            channels += growth_rate
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

# ========================= BASELINE NETWORK (M0) =========================
class DenseNet_Baseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        # Init conv (plain Conv + ReLU, no CDAN)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Dense blocks only (no CBAM)
        self.dense1 = DenseBlock(base_channels, growth_rate=32, layers=4)
        self.dense2 = DenseBlock(base_channels + 32*4, growth_rate=32, layers=4)

        # Final conv
        self.final_conv = nn.Conv2d(base_channels + 32*8, out_channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.final_conv(x)
        return torch.clamp(x, 0, 1)

# ========================= USAGE =========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNet_Baseline().to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x)
    print("Output shape:", out.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
