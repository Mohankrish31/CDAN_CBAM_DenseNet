import torch
from models.cdan_cbam_densenet import CDAN_CBAM_DenseNet
import os
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDAN_CBAM_DenseNet(self, in_channels=3, out_channels=3, base_channels=64).to(device)
# Training code here...
# After training, save weights:
os.makedirs("/content/saved_model", exist_ok=True)
model_path = "/content/saved_model/cdan_cbam_densenet.pth"
torch.save(model.state_dict(), model_path)
print("✅ Model weights saved at:", model_path)
