import matplotlib.pyplot as plt
# Dummy lists to store metrics (replace these with your actual metric values)
psnr_list = [25.3, 26.1, 24.8, 27.0]
ssim_list = [0.85, 0.87, 0.83, 0.88]
lpips_list = [0.25, 0.22, 0.28, 0.20]
ebcm_list = [0.12, 0.10, 0.14, 0.09]
# Plot metrics in subplots
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(psnr_list, marker='o')
plt.title('PSNR')
plt.subplot(2,2,2)
plt.plot(ssim_list, marker='o')
plt.title('SSIM')
plt.subplot(2,2,3)
plt.plot(lpips_list, marker='o')
plt.title('LPIPS')
plt.subplot(2,2,4)
plt.plot(ebcm_list, marker='o')
plt.title('EBCM')
plt.tight_layout()
plt.show()
