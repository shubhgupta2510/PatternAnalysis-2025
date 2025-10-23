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


def demonstrate_interpolation(model, test_images, save_dir='results'):
    """
    Demonstrate interpolation between two images.
    
    Args:
        model: Trained VQ-VAE model
        test_images: Test images
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Demonstrating Latent Space Interpolation")
    print("=" * 80)
    
    # Select two random images
    idx1, idx2 = 0, 7
    img1 = test_images[idx1:idx1+1]
    img2 = test_images[idx2:idx2+1]
    
    # Encode both images
    latent1 = model.encode(img1)
    latent2 = model.encode(img2)
    
    # Create interpolation
    num_steps = 8
    interpolations = []
    
    for alpha in np.linspace(0, 1, num_steps):
        # Linear interpolation in latent space
        interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
        
        # Decode
        decoded = model.decode(interpolated_latent)
        interpolations.append(decoded[0])
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 3))
    
    for i, (ax, img) in enumerate(zip(axes, interpolations)):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'α={i/(num_steps-1):.2f}', fontsize=10)
    
    plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / "interpolation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Interpolation plot saved to: {save_path}")
    plt.show()
    plt.close()

def create_ssim_histogram(ssim_scores, save_dir='results'):
    """
    Create histogram of SSIM scores.
    
    Args:
        ssim_scores: List of SSIM scores
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ssim_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.6, color='r', linestyle='--', linewidth=2, label='Threshold (0.6)')
    plt.axvline(x=np.mean(ssim_scores), color='g', linestyle='--', linewidth=2, 
                label=f'Mean ({np.mean(ssim_scores):.3f})')
    
    plt.xlabel('SSIM Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of SSIM Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = save_dir / "ssim_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SSIM histogram saved to: {save_path}")
    plt.show()
    plt.close()

def main():
    """Main prediction and demonstration function."""
    print("=" * 80)
    print("VQ-VAE Model Prediction and Demonstration")
    print("=" * 80)
    
    # Configuration
    BASE_DIR = "keras_slices_data"
    MODEL_PATH = "models/vqvae_best_*.h5"  # Use the best model
    RESULTS_DIR = "results"
    
    # Find the most recent model
    model_files = list(Path("models").glob("vqvae_best_*.h5"))
    if not model_files:
        print("Error: No trained model found!")
        print("Please train the model first using train.py")
        return
    
    # Use the most recent model
    model_path = sorted(model_files)[-1]
    
    # Load model
    model = load_trained_model(str(model_path))
    
    # Load test data
    print("\nLoading test data...")
    data_loader = HipMRIDataLoader(
        base_dir=BASE_DIR,
        target_size=(128, 128),
        batch_size=32
    )
    data_loader.load_data()
    _, _, test_dataset = data_loader.get_datasets()
    
    # Get sample images
    test_images = data_loader.get_sample_images(num_samples=16, split='test')
    
    # Evaluate model performance
    metrics = evaluate_model_performance(model, test_dataset, num_batches=10)
    
    # Demonstrate reconstruction
    demonstrate_reconstruction(model, test_images[:8], RESULTS_DIR)
    
    # Demonstrate latent space
    demonstrate_latent_space(model, test_images, RESULTS_DIR)
    
    # Demonstrate codebook analysis
    demonstrate_codebook_analysis(model, test_dataset, RESULTS_DIR)
    
    # Demonstrate interpolation
    demonstrate_interpolation(model, test_images, RESULTS_DIR)
    
    # Calculate SSIM for all test images in first few batches
    print("\nCalculating SSIM scores for histogram...")
    all_ssim_scores = []
    for i, batch in enumerate(test_dataset.take(20)):
        reconstructions = model.predict(batch, verbose=0)
        ssim_scores = calculate_ssim(batch.numpy(), reconstructions)
        all_ssim_scores.extend(ssim_scores)
    
    # Create SSIM histogram
    create_ssim_histogram(all_ssim_scores, RESULTS_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Model: {model_path.name}")
    print(f"Mean SSIM: {metrics['mean_ssim']:.4f}")
    print(f"SSIM Threshold Met: {'Yes' if metrics['mean_ssim'] >= 0.6 else 'No'}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()