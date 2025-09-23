import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # pip install lpips

# ========================= SSIM LOSS =========================
# You can use a simplified SSIM implementation
def ssim_loss(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean([2,3], keepdim=True)
    mu_y = target.mean([2,3], keepdim=True)
    
    sigma_x = ((pred - mu_x)**2).mean([2,3], keepdim=True)
    sigma_y = ((target - mu_y)**2).mean([2,3], keepdim=True)
    sigma_xy = ((pred - mu_x)*(target - mu_y)).mean([2,3], keepdim=True)
    
    ssim_n = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    
    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map.mean()), 0, 1)

# ========================= EDGE LOSS (EBCM) =========================
# EBCM: Edge-Based Contrast Measure
# We'll approximate it using Laplacian edge detection
def edge_loss(pred, target):
    laplace = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    laplace.weight = nn.Parameter(torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32))
    
    def edge(x):
        if x.shape[1] == 3:  # RGB -> convert to grayscale
            x = 0.2989*x[:,0:1,:,:] + 0.5870*x[:,1:2,:,:] + 0.1140*x[:,2:3,:,:]
        return laplace(x)
    
    pred_edge = edge(pred)
    target_edge = edge(target)
    return F.l1_loss(pred_edge, target_edge)

# ========================= COMBINED LOSS =========================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, delta=0.1, device='cuda'):
        """
        alpha -> MSE weight
        beta  -> SSIM weight
        gamma -> LPIPS weight
        delta -> Edge (EBCM) weight
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_l = ssim_loss(pred, target)
        lpips_l = self.lpips(pred, target).mean()
        ebcm_l = edge_loss(pred, target)
        
        total_loss = self.alpha*mse_loss + self.beta*ssim_l + self.gamma*lpips_l + self.delta*ebcm_l
        return total_loss

# ========================= USAGE =========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dummy input and target
    x = torch.randn(1,3,224,224).to(device)
    y = torch.randn(1,3,224,224).to(device)
    
    criterion = CombinedLoss(device=device)
    loss = criterion(x, y)
    
    print("Combined loss:", loss.item())
