import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import torch
from torch.nn import functional as F


# # SSIM Loss
# def ssim_loss(img1, img2):
#     return 1 - ssim(img1, img2, data_range=img2.max() - img2.min())

# # Mean Squared Error Loss
# def mse_loss(img1, img2):
#     return mean_squared_error(img1, img2)


def ssim_loss(img1, img2, window_size=11, data_range=1.0, size_average=True):
    """
    Compute Structural Similarity Index Measure (SSIM) loss between img1 and img2.

    Args:
        img1: Tensor of shape (batch_size, channels, height, width)
        img2: Tensor of shape (batch_size, channels, height, width)
        window_size: Size of the Gaussian filter and window
        data_range: The dynamic range of the images (usually 1.0 or 255)
        size_average: If True, compute the mean SSIM loss over the batch

    Returns:
        ssim_loss: SSIM loss value (or mean if size_average=True)
    """
    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    # Compute mean of img1 and img2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=0)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=0)

    # Compute variance and covariance
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=0) - mu1_mu2

    # Compute SSIM
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return torch.clamp((1 - ssim_map).mean() / compensation, 0, 1)
    else:
        return torch.clamp((1 - ssim_map) / compensation, 0, 1)


