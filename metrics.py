import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(img1, img2, max_val=255.0):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Parameters:
    - img1: The reconstructed image (tensor).
    - img2: The ground truth image (tensor).
    - max_val: The dynamic range of pixel values (default is 255).
    
    Returns:
    - PSNR value (float).
    """
    
    # Ensure both images have the same size
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")
    
    # Normalize the images to [0, max_val]
    img1 = img1.float()
    img2 = img2.float()
    
    min_img = min(img1.min(), img2.min())
    img1 = img1 - min_img
    img2 = img2 - min_img

    max_img = max(img1.max(), img2.max())
    img1 = max_val * img1 / max_img
    img2 = max_val * img2 / max_img

    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2, reduction='mean')
    
    if mse == 0:
        return float('inf')  # Return infinity if the images are identical

    # PSNR formula: 20 * log10(MAX_I) - 10 * log10(MSE)
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    
    return psnr.item()




def gaussian_kernel(window_size, sigma):
    """Generates a 3D Gaussian kernel."""
    kernel = torch.zeros((window_size, window_size, window_size))
    center = window_size // 2
    for x in range(window_size):
        for y in range(window_size):
            for z in range(window_size):
                radius_squared = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                # Corrected the line to use torch.tensor() for the computation
                kernel[x, y, z] = torch.exp(torch.tensor(-radius_squared / (2 * sigma ** 2)))
    kernel /= torch.sum(kernel)  # Normalize the kernel
    return kernel



def compute_ssim(img1, img2, sw=(2, 2, 2), K=(0.01, 0.03), L=255):
    """Computes the SSIM between two images."""
    
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")

    img1 = img1.float()
    img2 = img2.float()
    
    # Normalize the images to [0, 255]
    min_img = min(img1.min(), img2.min())
    img1 = img1 - min_img
    img2 = img2 - min_img

    max_img = max(img1.max(), img2.max())
    img1 = 255 * img1 / max_img
    img2 = 255 * img2 / max_img
    
    # Create a 3D Gaussian kernel
    window = gaussian_kernel(2 * sw[0] + 1, sigma=1.5).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0)  # Add channel dimensions for convolution

    # SSIM constants
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    # Mean calculations
    mu1 = F.conv3d(img1, window, padding=sw)
    mu2 = F.conv3d(img2, window, padding=sw)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance and covariance calculations
    sigma1_sq = F.conv3d(img1 * img1, window, padding=sw) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=sw) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=sw) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Mean SSIM
    mssim = ssim_map.mean()

    return mssim, ssim_map





