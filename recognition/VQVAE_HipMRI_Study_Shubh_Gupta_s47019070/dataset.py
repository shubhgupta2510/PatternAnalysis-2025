"""
Data Loader for HipMRI Study Dataset
Author: Shubh Gupta (s47019070)
Description: Functions for loading and preprocessing 2D MRI slices
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
import os
import glob
from pathlib import Path


def load_nifti_slice(file_path):
    """
    Load a single NIfTI file and extract the 2D slice.
    
    Args:
        file_path: Path to .nii.gz file
        
    Returns:
        2D numpy array containing the MRI slice
    """
    try:
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        
        # Handle different dimensions
        if img_data.ndim == 2:
            slice_2d = img_data
        elif img_data.ndim == 3:
            # Take middle slice if 3D
            slice_2d = img_data[:, :, img_data.shape[2] // 2]
        else:
            raise ValueError(f"Unexpected number of dimensions: {img_data.ndim}")
        
        return slice_2d
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_image(image):
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image
    """
    # Handle case where image might be all zeros
    if np.max(image) == np.min(image):
        return np.zeros_like(image)
    
    # Min-max normalization
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

