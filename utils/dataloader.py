import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# ================= Dataset =================
class cvccolondbsplitDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
               os.path.exists(os.path.join(high_dir, f))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.image_names[idx])
        high_path = os.path.join(self.high_dir, self.image_names[idx])

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


# ================= Hyperparameters =================
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 100
batch_size = 8
early_stopping_patience = 10
resize_dim = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Paths =================
train_high_dir = "/content/cvccolondbsplit/train/high"
train_low_dir = "/content/cvccolondbsplit/train/low"
val_high_dir = "/content/cvccolondbsplit/val/high"
val_low_dir = "/content/cvccolondbsplit/val/low"

# ================= Use your precomputed mean & std =================
dataset_mean = [0.31112134, 0.18268488, 0.10600837]
dataset_std  = [0.21178198, 0.1397511, 0.0843721]

print(f"Dataset Mean: {dataset_mean}")
print(f"Dataset Std: {dataset_std}")

# ================= Transforms with Normalization =================
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])

# ================= Create Dataset Instances =================
train_dataset = cvccolondbsplitDataset(train_low_dir, train_high_dir, transform=transform)
val_dataset = cvccolondbsplitDataset(val_low_dir, val_high_dir, transform=transform)

# ================= Create DataLoaders =================
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"First batch shape: {next(iter(train_loader))[0].shape}")

# ================= Denormalization Function =================
def denormalize(tensor, mean, std):
    """
    Correctly denormalizes a tensor using the dataset-specific mean and std.
    tensor: [B, C, H, W]
    mean, std: lists or tensors of length 3
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

# ================= Example usage =================
# batch_low, batch_high = next(iter(train_loader))
# batch_low_denorm = denormalize(batch_low, dataset_mean, dataset_std)
