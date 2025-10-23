"""
Utility Functions for VQ-VAE Project
Author: Shubh Gupta (s47019070)
Description: Helper functions for visualisation, metrics, and analysis
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