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

def create_tf_dataset(images, batch_size=32, shuffle=True, buffer_size=1000):
    """
    Create a TensorFlow dataset from image arrays.
    
    Args:
        images: Numpy array of images
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        
    Returns:
        tf.data.Dataset object
    """
    dataset = tf.data.Dataset.from_tensor_slices(images)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

class HipMRIDataLoader:
    """
    Main data loader class for HipMRI dataset.
    Handles loading train, validation, and test splits.
    """
    
    def __init__(self, base_dir, target_size=(128, 128), batch_size=32):
        """
        Args:
            base_dir: Base directory containing keras_slices_data folder
            target_size: Target size for images
            batch_size: Batch size for datasets
        """
        self.base_dir = Path(base_dir)
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Define data directories
        self.train_dir = self.base_dir / "keras_slices_train"
        self.val_dir = self.base_dir / "keras_slices_validate"
        self.test_dir = self.base_dir / "keras_slices_test"
        
        # Check if directories exist
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {dir_path}")
        
        self.train_images = None
        self.val_images = None
        self.test_images = None
        
    def load_data(self, max_train=None, max_val=None, max_test=None):
        """
        Load all datasets.
        
        Args:
            max_train: Max training images to load
            max_val: Max validation images to load
            max_test: Max test images to load
        """
        print("Loading training data...")
        self.train_images = load_dataset_from_directory(
            str(self.train_dir), self.target_size, max_train
        )
        
        print("Loading validation data...")
        self.val_images = load_dataset_from_directory(
            str(self.val_dir), self.target_size, max_val
        )
        
        print("Loading test data...")
        self.test_images = load_dataset_from_directory(
            str(self.test_dir), self.target_size, max_test
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Training samples: {len(self.train_images)}")
        print(f"Validation samples: {len(self.val_images)}")
        print(f"Test samples: {len(self.test_images)}")
        
    def get_datasets(self, shuffle_train=True):
        """
        Get TensorFlow datasets for training, validation, and testing.
        
        Args:
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.train_images is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        train_dataset = create_tf_dataset(
            self.train_images, 
            self.batch_size, 
            shuffle=shuffle_train
        )
        
        val_dataset = create_tf_dataset(
            self.val_images, 
            self.batch_size, 
            shuffle=False
        )
        
        test_dataset = create_tf_dataset(
            self.test_images, 
            self.batch_size, 
            shuffle=False
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_sample_images(self, num_samples=5, split='test'):
        """
        Get sample images for visualization.
        
        Args:
            num_samples: Number of samples to return
            split: Which split to sample from ('train', 'val', 'test')
            
        Returns:
            Array of sample images
        """
        if split == 'train':
            images = self.train_images
        elif split == 'val':
            images = self.val_images
        else:
            images = self.test_images
        
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        return images[indices]

def get_data_statistics(data_dir):
    """
    Calculate statistics about the dataset.
    
    Args:
        data_dir: Directory containing .nii.gz files
        
    Returns:
        Dictionary with dataset statistics
    """
    files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    
    if len(files) == 0:
        return {"num_files": 0}
    
    # Load a sample to get image dimensions
    sample_img = load_nifti_slice(files[0])
    
    stats = {
        "num_files": len(files),
        "original_shape": sample_img.shape if sample_img is not None else None,
    }
    
    return stats
