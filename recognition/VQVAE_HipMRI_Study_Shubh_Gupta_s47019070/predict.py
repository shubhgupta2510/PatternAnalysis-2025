"""
Prediction and Inference Script for VQ-VAE
Author: Shubh Gupta (s47019070)
Description: Load trained model and demonstrate usage with visualisations
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import custom modules
from modules import VQVAE, VectorQuantizer, Encoder, Decoder, ResidualBlock, build_vqvae
from dataset import HipMRIDataLoader
from utils import (
    calculate_ssim, 
    plot_reconstructions,
    plot_latent_space_visualization,
    plot_codebook_usage,
    generate_random_samples,
    plot_generated_samples
)

def load_trained_model(model_path):
    """
    Load a trained VQ-VAE model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    print(f"Loading model from: {model_path}")
    
    # Try to load the complete model first
    try:
        # Define custom objects for loading
        custom_objects = {
            'VQVAE': VQVAE,
            'VectorQuantizer': VectorQuantizer,
            'Encoder': Encoder,
            'Decoder': Decoder,
            'ResidualBlock': ResidualBlock,
        }
        
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Could not load complete model: {e}")
        print("Attempting to rebuild model and load weights...")
        
        # Rebuild the model architecture
        model = build_vqvae(
            input_shape=(128, 128, 1),
            latent_dim=64,
            num_embeddings=512,
            num_residual_blocks=2,
            commitment_cost=0.25
        )
        
        # Load weights
        try:
            model.load_weights(model_path)
            print("Model weights loaded successfully!")
            return model
        except Exception as e2:
            print(f"Error loading weights: {e2}")
            raise ValueError(f"Could not load model from {model_path}")

def evaluate_model_performance(model, test_dataset, num_batches=10):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained VQ-VAE model
        test_dataset: Test dataset
        num_batches: Number of batches to evaluate
        
    Returns:
        Dictionary of performance metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating Model Performance")
    print("=" * 80)
    
    all_ssim_scores = []
    reconstruction_errors = []
    
    for i, batch in enumerate(test_dataset.take(num_batches)):
        # Get reconstructions
        reconstructions = model.predict(batch, verbose=0)
        
        # Calculate SSIM
        ssim_scores = calculate_ssim(batch.numpy(), reconstructions)
        all_ssim_scores.extend(ssim_scores)
        
        # Calculate reconstruction error (MSE)
        mse = np.mean((batch.numpy() - reconstructions) ** 2, axis=(1, 2, 3))
        reconstruction_errors.extend(mse)
    
    # Calculate statistics
    metrics = {
        'mean_ssim': np.mean(all_ssim_scores),
        'std_ssim': np.std(all_ssim_scores),
        'min_ssim': np.min(all_ssim_scores),
        'max_ssim': np.max(all_ssim_scores),
        'mean_mse': np.mean(reconstruction_errors),
        'std_mse': np.std(reconstruction_errors),
    }
    
    # Print results
    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"Mean SSIM:       {metrics['mean_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
    print(f"SSIM Range:      [{metrics['min_ssim']:.4f}, {metrics['max_ssim']:.4f}]")
    print(f"Mean MSE:        {metrics['mean_mse']:.6f} ± {metrics['std_mse']:.6f}")
    print("-" * 80)
    
    # Check if SSIM threshold is met
    if metrics['mean_ssim'] >= 0.6:
        print("✓ Model meets SSIM threshold of 0.6!")
    else:
        print("✗ Model does not meet SSIM threshold of 0.6")
    
    print("=" * 80)
    
    return metrics

def demonstrate_reconstruction(model, test_images, save_dir='results'):
    """
    Demonstrate image reconstruction capabilities.
    
    Args:
        model: Trained VQ-VAE model
        test_images: Test images
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Demonstrating Image Reconstruction")
    print("=" * 80)
    
    # Reconstruct images
    reconstructions = model.predict(test_images, verbose=0)
    
    # Calculate SSIM for each image
    ssim_scores = calculate_ssim(test_images, reconstructions)
    
    # Plot reconstructions
    save_path = save_dir / "reconstruction_examples.png"
    plot_reconstructions(model, test_images, str(save_path), num_images=8)
    
    # Print individual SSIM scores
    print("\nIndividual SSIM Scores:")
    print("-" * 80)
    for i, score in enumerate(ssim_scores[:8]):
        print(f"Image {i+1}: SSIM = {score:.4f}")
    print("-" * 80)

def demonstrate_latent_space(model, test_images, save_dir='results'):
    """
    Visualize latent space representations.
    
    Args:
        model: Trained VQ-VAE model
        test_images: Test images
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Visualizing Latent Space")
    print("=" * 80)
    
    save_path = save_dir / "latent_space_visualization.png"
    plot_latent_space_visualization(model, test_images, str(save_path))

def demonstrate_codebook_analysis(model, test_dataset, save_dir='results'):
    """
    Analyze codebook usage.
    
    Args:
        model: Trained VQ-VAE model
        test_dataset: Test dataset
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Analyzing Codebook Usage")
    print("=" * 80)
    
    save_path = save_dir / "codebook_usage.png"
    plot_codebook_usage(model, test_dataset, str(save_path))

