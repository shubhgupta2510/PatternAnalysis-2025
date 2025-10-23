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
