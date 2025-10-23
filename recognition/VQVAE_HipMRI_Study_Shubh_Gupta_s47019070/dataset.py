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

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess a single image: resize and normalize.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Normalize
    image = normalize_image(image)
    
    # Convert to tensor and resize
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension
    image = tf.image.resize(image, target_size)
    
    return image

def load_dataset_from_directory(data_dir, target_size=(128, 128), max_files=None):
    """
    Load all NIfTI files from a directory.
    
    Args:
        data_dir: Directory containing .nii.gz files
        target_size: Target size for images
        max_files: Maximum number of files to load (None for all)
        
    Returns:
        Numpy array of preprocessed images
    """
    # Get all .nii.gz files
    file_pattern = os.path.join(data_dir, "*.nii.gz")
    files = sorted(glob.glob(file_pattern))
    
    if max_files is not None:
        files = files[:max_files]
    
    print(f"Loading {len(files)} files from {data_dir}")
    
    images = []
    for i, file_path in enumerate(files):
        if i % 100 == 0:
            print(f"Loading file {i}/{len(files)}")
        
        img = load_nifti_slice(file_path)
        if img is not None:
            img = preprocess_image(img, target_size)
            images.append(img.numpy())
    
    if len(images) == 0:
        raise ValueError(f"No valid images found in {data_dir}")
    
    return np.array(images)

