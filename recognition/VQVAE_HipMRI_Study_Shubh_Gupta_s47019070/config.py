"""
Configuration file for VQ-VAE HipMRI Study
Author: Shubh Gupta (s47019070)
Description: Central configuration for hyperparameters and paths
"""

import os
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "keras_slices_data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Image dimensions
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Data splits
TRAIN_DIR = DATA_DIR / "keras_slices_train"
VAL_DIR = DATA_DIR / "keras_slices_validate"
TEST_DIR = DATA_DIR / "keras_slices_test"

# VQ-VAE hyperparameters
LATENT_DIM = 64                 # Dimension of latent space
NUM_EMBEDDINGS = 512            # Size of codebook
NUM_RESIDUAL_BLOCKS = 2         # Number of residual blocks in encoder/decoder
COMMITMENT_COST = 0.25          # Weight for commitment loss

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15

# Optimizer settings
OPTIMIZER = "adam"
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

# Early stopping
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# SSIM threshold for success
SSIM_THRESHOLD = 0.6

# Number of samples for visualization
NUM_VISUALIZATION_SAMPLES = 8
NUM_INTERPOLATION_STEPS = 8

# Random seeds
RANDOM_SEED = 42

# TensorFlow settings
TF_DETERMINISTIC = True
TF_CUDNN_DETERMINISTIC = True

# Verbosity levels
VERBOSE_TRAINING = 1
VERBOSE_EVALUATION = 1

# TensorBoard
TENSORBOARD_UPDATE_FREQ = "epoch"
TENSORBOARD_HISTOGRAM_FREQ = 1

# Plot settings
PLOT_DPI = 300
PLOT_FORMAT = "png"
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_MEDIUM = (14, 8)
FIGURE_SIZE_LARGE = (18, 10)

# Model saving
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_ONLY = False
MONITOR_METRIC = "val_total_loss"
MONITOR_MODE = "min"