import matplotlib.pyplot as plt
# Dummy lists to store metrics
mse_list, ssim_list, lpips_list, ebcm_list, total_list = [], [], [], [], []
# Example training loop (replace with real data/model)
for epoch in range(100):
    total_loss, mse_l, ssim_l, lpips_l, ebcm_l = criterion(x, y)
    # store values
    total_list.append(total_loss.item())
    mse_list.append(mse_l)
    ssim_list.append(ssim_l)
    lpips_list.append(lpips_l)
    ebcm_list.append(ebcm_l)
# Plot losses separately
plt.figure(figsize=(12,8))
plt.subplot(3,2,1)
plt.plot(total_list, marker='o'); plt.title('Total Loss')
plt.subplot(3,2,2)
plt.plot(mse_list, marker='o'); plt.title('MSE Loss')
plt.subplot(3,2,3)
plt.plot(ssim_list, marker='o'); plt.title('SSIM Loss')
plt.subplot(3,2,4)
plt.plot(lpips_list, marker='o'); plt.title('LPIPS Loss')
plt.subplot(3,2,5)
plt.plot(ebcm_list, marker='o'); plt.title('EBCM Loss')
plt.tight_layout()
plt.show()
