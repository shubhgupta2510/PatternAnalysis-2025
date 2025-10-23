"""
Utility Functions for VQ-VAE Project
Author: Shubh Gupta (s47019070)
Description: Helper functions for visualisation, metrics, and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


def calculate_ssim(original_images, reconstructed_images):
    """
    Calculate SSIM between original and reconstructed images.
    
    Args:
        original_images: Original images array
        reconstructed_images: Reconstructed images array
        
    Returns:
        List of SSIM scores
    """
    ssim_scores = []
    
    for orig, recon in zip(original_images, reconstructed_images):
        # Remove channel dimension for SSIM calculation
        orig_2d = orig.squeeze()
        recon_2d = recon.squeeze()
        
        # Calculate SSIM
        score = ssim(orig_2d, recon_2d, data_range=1.0)
        ssim_scores.append(score)
    
    return ssim_scores
