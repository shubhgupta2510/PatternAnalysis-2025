"""
Training Script for VQ-VAE on HipMRI Dataset
Author: Shubh Gupta (s47019070)
Description: Train, validate, and test the VQ-VAE model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from modules import build_vqvae
from dataset import HipMRIDataLoader
from utils import calculate_ssim, plot_training_history, plot_reconstructions


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SSIMMetric(keras.metrics.Metric):
    """Custom metric to calculate SSIM during training."""
    
    def __init__(self, name='ssim', **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.ssim_sum = self.add_weight(name='ssim_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        self.ssim_sum.assign_add(ssim_value)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.ssim_sum / self.count
    
    def reset_state(self):
        self.ssim_sum.assign(0.0)
        self.count.assign(0.0)
        

def train_vqvae(base_dir,
                input_shape=(128, 128, 1),
                latent_dim=64,
                num_embeddings=512,
                num_residual_blocks=2,
                commitment_cost=0.25,
                batch_size=32,
                epochs=100,
                learning_rate=1e-3,
                save_dir='models',
                log_dir='logs'):
    """
    Train the VQ-VAE model.
    
    Args:
        base_dir: Base directory containing the data
        input_shape: Shape of input images
        latent_dim: Dimension of latent space
        num_embeddings: Size of codebook
        num_residual_blocks: Number of residual blocks
        commitment_cost: Weight for commitment loss
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save models
        log_dir: Directory to save logs
    
    Returns:
        Trained model and training history
    """
    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("VQ-VAE Training for HipMRI Study")
    print("=" * 80)
    print(f"Input shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Codebook size: {num_embeddings}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80)
    
    # Load data
    print("\nLoading dataset...")
    data_loader = HipMRIDataLoader(
        base_dir=base_dir,
        target_size=input_shape[:2],
        batch_size=batch_size
    )
    data_loader.load_data()
    
    train_dataset, val_dataset, test_dataset = data_loader.get_datasets()
    
    # Build model
    print("\nBuilding VQ-VAE model...")
    model = build_vqvae(
        input_shape=input_shape,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        num_residual_blocks=num_residual_blocks,
        commitment_cost=commitment_cost
    )
    
    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Print model information
    print("\nModel Architecture:")
    print("-" * 80)
    print(f"Encoder: {model.encoder.__class__.__name__}")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Residual blocks: {num_residual_blocks}")
    print(f"Vector Quantizer:")
    print(f"  - Codebook size: {num_embeddings}")
    print(f"  - Embedding dimension: {latent_dim}")
    print(f"  - Commitment cost: {commitment_cost}")
    print(f"Decoder: {model.decoder.__class__.__name__}")
    print(f"  - Residual blocks: {num_residual_blocks}")
    print("-" * 80)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    checkpoint_path = save_dir / f"vqvae_checkpoint_{timestamp}.h5"
    best_model_path = save_dir / f"vqvae_best_{timestamp}.h5"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_total_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_total_loss',
            mode='min',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir / timestamp),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    print("=" * 80)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = save_dir / f"vqvae_final_{timestamp}.h5"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"Test Results: {dict(zip(model.metrics_names, test_results))}")
    
    # Calculate SSIM on test set
    print("\nCalculating SSIM on test samples...")
    ssim_scores = []
    num_test_batches = 5  # Evaluate on first 5 batches
    
    for i, batch in enumerate(test_dataset.take(num_test_batches)):
        reconstructions = model.predict(batch, verbose=0)
        batch_ssim = calculate_ssim(batch.numpy(), reconstructions)
        ssim_scores.extend(batch_ssim)
    
    mean_ssim = np.mean(ssim_scores)
    print(f"Mean SSIM on test set: {mean_ssim:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    history_plot_path = log_dir / f"training_history_{timestamp}.png"
    plot_training_history(history, str(history_plot_path))
    
    # Plot sample reconstructions
    print("Plotting sample reconstructions...")
    sample_images = data_loader.get_sample_images(num_samples=8, split='test')
    recon_plot_path = log_dir / f"reconstructions_{timestamp}.png"
    plot_reconstructions(model, sample_images, str(recon_plot_path))
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Plots saved to: {log_dir}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print("=" * 80)
    
    return model, history