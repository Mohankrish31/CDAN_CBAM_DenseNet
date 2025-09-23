import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# === Dataset ===
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


# --- Function to calculate mean and std ---
def get_mean_std(dataset_path, resize_dim):
    print("⏳ Calculating mean and standard deviation of the training dataset...")
    temp_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor()
    ])
    
    high_dir = os.path.join(dataset_path, 'high')
    image_names = [f for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for fname in tqdm(image_names):
        img_path = os.path.join(high_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = temp_transform(img)
        
        mean += img_tensor.mean(dim=[1, 2])
        std += img_tensor.std(dim=[1, 2])
        num_samples += 1

    mean /= num_samples
    std /= num_samples

    print("✅ Mean and Std calculated.")
    return mean.tolist(), std.tolist()


# ---------- Hyperparams ----------
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 100
batch_size = 8
early_stopping_patience = 10
resize_dim = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
train_high_dir = "/content/cvccolondbsplit/train/high"
train_low_dir = "/content/cvccolondbsplit/train/low"
val_high_dir = "/content/cvccolondbsplit/val/high"
val_low_dir = "/content/cvccolondbsplit/val/low"

# --- Calculate dataset-specific mean and std ---
# This is a one-time calculation.
dataset_mean, dataset_std = get_mean_std(os.path.dirname(train_high_dir), resize_dim)

print(f"Dataset Mean: {dataset_mean}")
print(f"Dataset Std: {dataset_std}")

# === Define Transforms with Normalization ===
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])

# === Create Dataset Instances ===
train_dataset = cvccolondbsplitDataset(train_low_dir, train_high_dir, transform=transform)
val_dataset = cvccolondbsplitDataset(val_low_dir, val_high_dir, transform=transform)

# === Create DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"First batch shape: {next(iter(train_loader))[0].shape}")
