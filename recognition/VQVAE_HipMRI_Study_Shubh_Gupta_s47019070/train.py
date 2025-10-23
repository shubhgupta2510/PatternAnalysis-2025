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