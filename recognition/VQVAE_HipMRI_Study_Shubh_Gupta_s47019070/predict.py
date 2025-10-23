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
