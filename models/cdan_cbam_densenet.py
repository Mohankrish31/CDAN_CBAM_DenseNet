import torch
import torch.nn as nn
import torch.nn.functional as F
# ========================= CBAM MODULE =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out) * x
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
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
# ========================= CDAN BASE BLOCK =========================
class CDANBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x
# ========================= CDAN + CBAM + DENSE NETWORK =========================
class CDAN_CBAM_DenseNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.init_conv = CDANBlock(in_channels, base_channels)
        self.dense1 = DenseBlock(base_channels, growth_rate=32, layers=4)
        self.cbam1 = CBAM(base_channels + 32*4)
        self.dense2 = DenseBlock(base_channels + 32*4, growth_rate=32, layers=4)
        self.cbam2 = CBAM(base_channels + 32*8)
        self.final_conv = nn.Conv2d(base_channels + 32*8, out_channels, 1)
    def forward(self, x):
        x = self.init_conv(x)
        x = self.dense1(x)
        x = self.cbam1(x)
        x = self.dense2(x)
        x = self.cbam2(x)
        x = self.final_conv(x)
        return torch.clamp(x, 0, 1)
# ========================= USAGE =========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CDAN_CBAM_DenseNet().to(device)
    # dummy input
    x = torch.randn(1,3,224,224).to(device)
    out = model(x)
    print("Output shape:", out.shape)
