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


def plot_reconstructions(model, images, save_path=None, num_images=8):
    """
    Plot original images and their reconstructions.
    
    Args:
        model: Trained VQ-VAE model
        images: Original images
        save_path: Path to save the plot
        num_images: Number of images to plot
    """
    num_images = min(num_images, len(images))
    
    # Get reconstructions
    reconstructions = model.predict(images[:num_images], verbose=0)
    
    # Calculate SSIM for each pair
    ssim_scores = calculate_ssim(images[:num_images], reconstructions)
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(2.5 * num_images, 5))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f'Reconstruction\nSSIM: {ssim_scores[i]:.3f}', 
                               fontsize=12, fontweight='bold')
        else:
            axes[1, i].set_title(f'SSIM: {ssim_scores[i]:.3f}', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction plot saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    return ssim_scores

def plot_latent_space_visualization(model, images, save_path=None):
    """
    Visualize the latent space representations.
    
    Args:
        model: Trained VQ-VAE model
        images: Input images
        save_path: Path to save the plot
    """
    # Encode images
    latents = model.encode(images[:16])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(16, len(latents))):
        # Visualize mean across channels
        latent_viz = np.mean(latents[i], axis=-1)
        
        im = axes[i].imshow(latent_viz, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Latent {i+1}', fontsize=10)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Latent Space Representations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_codebook_usage(model, dataset, save_path=None):
    """
    Analyze and plot codebook usage statistics.
    
    Args:
        model: Trained VQ-VAE model
        dataset: TensorFlow dataset
        save_path: Path to save the plot
    """
    codebook_counts = np.zeros(model.num_embeddings)
    
    # Count codebook usage
    for batch in dataset.take(10):  # Take 10 batches for analysis
        encoded = model.encoder(batch)
        
        # Get codebook indices
        flattened = tf.reshape(encoded, [-1, model.latent_dim])
        indices = model.vq_layer.get_code_indices(flattened)
        
        # Count occurrences
        unique, counts = np.unique(indices.numpy(), return_counts=True)
        for idx, count in zip(unique, counts):
            codebook_counts[idx] += count
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].bar(range(len(codebook_counts)), codebook_counts)
    axes[0].set_xlabel('Codebook Index', fontsize=12)
    axes[0].set_ylabel('Usage Count', fontsize=12)
    axes[0].set_title('Codebook Usage Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Statistics
    used_codes = np.sum(codebook_counts > 0)
    usage_percentage = (used_codes / len(codebook_counts)) * 100
    
    stats_text = f"Total Codes: {len(codebook_counts)}\n"
    stats_text += f"Used Codes: {used_codes}\n"
    stats_text += f"Usage: {usage_percentage:.1f}%\n"
    stats_text += f"Mean Count: {np.mean(codebook_counts):.1f}\n"
    stats_text += f"Max Count: {np.max(codebook_counts):.0f}"
    
    axes[1].text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
                verticalalignment='center')
    axes[1].axis('off')
    axes[1].set_title('Codebook Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Codebook usage plot saved to: {save_path}")
    
    plt.show()
    plt.close()

def save_model_config(config, save_path):
    """
    Save model configuration to a text file.
    
    Args:
        config: Dictionary of configuration parameters
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as f:
        f.write("VQ-VAE Model Configuration\n")
        f.write("=" * 50 + "\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Configuration saved to: {save_path}")


def compare_models(models_dict, test_images, save_path=None):
    """
    Compare multiple models on the same test images.
    
    Args:
        models_dict: Dictionary of {model_name: model} pairs
        test_images: Test images to use
        save_path: Path to save the comparison plot
    """
    num_models = len(models_dict)
    num_images = min(5, len(test_images))
    
    fig, axes = plt.subplots(num_models + 1, num_images, 
                            figsize=(2.5 * num_images, 2.5 * (num_models + 1)))
    
    # Plot original images
    for j in range(num_images):
        axes[0, j].imshow(test_images[j].squeeze(), cmap='gray')
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_ylabel('Original', fontsize=10, fontweight='bold')
    
    # Plot reconstructions for each model
    for i, (model_name, model) in enumerate(models_dict.items(), 1):
        reconstructions = model.predict(test_images[:num_images], verbose=0)
        ssim_scores = calculate_ssim(test_images[:num_images], reconstructions)
        
        for j in range(num_images):
            axes[i, j].imshow(reconstructions[j].squeeze(), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'SSIM: {ssim_scores[j]:.3f}', fontsize=8)
            
            if j == 0:
                axes[i, j].set_ylabel(model_name, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def generate_random_samples(model, num_samples=16, latent_shape=(32, 32, 64)):
    """
    Generate random samples by decoding random latent codes.
    
    Args:
        model: Trained VQ-VAE model
        num_samples: Number of samples to generate
        latent_shape: Shape of latent space
        
    Returns:
        Generated images
    """
    # Sample random latent codes
    random_latents = tf.random.normal((num_samples,) + latent_shape)
    
    # Quantize
    quantized = model.vq_layer(random_latents)
    
    # Decode
    generated = model.decoder(quantized)
    
    return generated.numpy()


def plot_generated_samples(generated_images, save_path=None):
    """
    Plot generated samples.
    
    Args:
        generated_images: Array of generated images
        save_path: Path to save the plot
    """
    num_images = min(16, len(generated_images))
    grid_size = int(np.sqrt(num_images))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Generated Samples from VQ-VAE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generated samples plot saved to: {save_path}")
    
    plt.show()
    plt.close()
