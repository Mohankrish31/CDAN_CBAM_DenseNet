import os
base_dir = "/content/drive/MyDrive/Colon_Enhanced"
os.makedirs(os.path.join(base_dir, "train_enhanced"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "val_enhanced"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "test_enhanced"), exist_ok=True)
