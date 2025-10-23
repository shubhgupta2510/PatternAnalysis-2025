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

def plot_training_history(history, save_path=None):
    """
    Plot training history including losses.
    
    Args:
        history: Keras History object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot total loss
    axes[0].plot(history.history['total_loss'], label='Train Total Loss', linewidth=2)
    if 'val_total_loss' in history.history:
        axes[0].plot(history.history['val_total_loss'], label='Val Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot reconstruction loss
    axes[1].plot(history.history['reconstruction_loss'], label='Train Recon Loss', linewidth=2)
    if 'val_reconstruction_loss' in history.history:
        axes[1].plot(history.history['val_reconstruction_loss'], label='Val Recon Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot VQ loss
    axes[2].plot(history.history['vq_loss'], label='Train VQ Loss', linewidth=2)
    if 'val_vq_loss' in history.history:
        axes[2].plot(history.history['val_vq_loss'], label='Val VQ Loss', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('VQ Loss', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()
    plt.close()

