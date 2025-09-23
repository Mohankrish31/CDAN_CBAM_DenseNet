import torch
import torch.nn.functional as F
import lpips
import math

# ========================= METRICS FUNCTIONS =========================

# 1. PSNR
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

# 2. SSIM (simplified)
def ssim_metric(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean([2,3], keepdim=True)
    mu_y = target.mean([2,3], keepdim=True)
    sigma_x = ((pred - mu_x)**2).mean([2,3], keepdim=True)
    sigma_y = ((target - mu_y)**2).mean([2,3], keepdim=True)
    sigma_xy = ((pred - mu_x)*(target - mu_y)).mean([2,3], keepdim=True)
    ssim_n = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    ssim_map = ssim_n / ssim_d
    return ssim_map.mean().item()

# 3. Edge-Based Contrast Measure (EBCM)
def ebcm_metric(pred, target):
    laplace = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32)
    if pred.is_cuda:
        laplace = laplace.to(pred.device)

    def edge(x):
        if x.shape[1] == 3:
            x = 0.2989*x[:,0:1,:,:] + 0.5870*x[:,1:2,:,:] + 0.1140*x[:,2:3,:,:]
        return F.conv2d(x, laplace, padding=1)

    pred_edge = edge(pred)
    target_edge = edge(target)
    return F.l1_loss(pred_edge, target_edge).item()

# 4. LPIPS (requires lpips package)
lpips_model = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    lpips_model = lpips_model.cuda()

def lpips_metric(pred, target):
    return lpips_model(pred, target).mean().item()

# ========================= USAGE EXAMPLE =========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dummy batch of predicted and ground truth images
    pred = torch.rand(2,3,224,224).to(device)  # predicted images
    target = torch.rand(2,3,224,224).to(device)  # ground truth

    psnr_val = psnr(pred, target)
    ssim_val = ssim_metric(pred, target)
    ebcm_val = ebcm_metric(pred, target)
    lpips_val = lpips_metric(pred, target)

    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"EBCM: {ebcm_val:.4f}")
    print(f"LPIPS: {lpips_val:.4f}")
 # === Run the function === #
  high_dir = "/content/cvccolondbsplit/test/high"
 enhanced_dir = "/content/outputs/test_enhanced"
 evaluate_metrics_individual(high_dir, enhanced_dir)
