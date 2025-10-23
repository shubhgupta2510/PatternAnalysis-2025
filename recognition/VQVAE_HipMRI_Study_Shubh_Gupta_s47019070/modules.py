
"""
VQ-VAE Model Components for HipMRI Study
Author: Shubh Gupta (s47019070)
Description: Implementation of Vector Quantized Variational AutoEncoder components
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class VectorQuantizer(layers.Layer):
    """
    Vector Quantization layer for VQ-VAE.
    Maps continuous encoder outputs to discrete codebook vectors.
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, **kwargs):
        """
        Args:
            num_embeddings: Number of vectors in the codebook
            embedding_dim: Dimension of each codebook vector
            commitment_cost: Weight for commitment loss
        """
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialise codebook embeddings with better initialisation
        w_init = tf.random_uniform_initializer(minval=-1.0 / self.num_embeddings, 
                                               maxval=1.0 / self.num_embeddings)
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), 
                                dtype='float32'),
            trainable=True,
            name='embeddings'
        )
        
    def call(self, inputs):
        """
        Forward pass through vector quantization.
        
        Args:
            inputs: Encoder outputs of shape [batch, height, width, channels]
            
        Returns:
            quantized: Quantized vectors
        """
        # Flatten input to [batch*height*width, embedding_dim]
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, self.embedding_dim])
        
        # Calculate distances to codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        # Quantize
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        
        # Calculate VQ losses - properly scaled
        # Codebook loss: trains embeddings to be close to encoder outputs
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        
        # Commitment loss: trains encoder to commit to codebook vectors
        commitment_loss = self.commitment_cost * tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        
        # Total VQ loss
        vq_loss = codebook_loss + commitment_loss
        
        # Straight-through estimator for gradients
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        
        # Add loss to layer
        self.add_loss(vq_loss)
        
        return quantized
    
    def get_code_indices(self, flattened_inputs):
        """
        Calculate nearest codebook indices for input vectors.
        """
        # Calculate L2 distances
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        
        # Get indices of nearest embeddings
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class ResidualBlock(layers.Layer):
    """
    Residual block with two convolutional layers and skip connection.
    """
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.activation = layers.Activation('relu')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return self.activation(x + inputs)
    
class Encoder(layers.Layer):
    """
    Encoder network for VQ-VAE.
    Downsamples input images to latent representation.
    """
    
    def __init__(self, latent_dim=64, num_residual_blocks=2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Downsampling layers with simpler architecture
        self.conv1 = layers.Conv2D(32, 4, strides=2, padding='same', 
                                   kernel_initializer='he_normal', activation='relu')
        self.conv2 = layers.Conv2D(64, 4, strides=2, padding='same',
                                   kernel_initializer='he_normal', activation='relu')
        self.conv3 = layers.Conv2D(latent_dim, 3, padding='same',
                                   kernel_initializer='he_normal')
        
        # Residual blocks
        self.residual_blocks = [
            ResidualBlock(latent_dim) for _ in range(num_residual_blocks)
        ]
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)  # No activation here - let VQ layer handle it
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            
        return x


class Decoder(layers.Layer):
    """
    Decoder network for VQ-VAE.
    Upsamples quantized latent vectors back to image space.
    """
    
    def __init__(self, num_residual_blocks=2, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        # Residual blocks
        self.residual_blocks = [
            ResidualBlock(64) for _ in range(num_residual_blocks)
        ]
        
        # Upsampling layers
        self.conv1 = layers.Conv2D(64, 3, padding='same',
                                   kernel_initializer='he_normal', activation='relu')
        self.upsample1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                                                kernel_initializer='he_normal', activation='relu')
        self.upsample2 = layers.Conv2DTranspose(32, 4, strides=2, padding='same',
                                                kernel_initializer='he_normal', activation='relu')
        self.conv_out = layers.Conv2D(1, 3, padding='same', activation='sigmoid',
                                      kernel_initializer='glorot_uniform')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv_out(x)
        
        return x


class VQVAE(keras.Model):
    """
    Complete VQ-VAE model combining encoder, vector quantizer, and decoder.
    """
    
    def __init__(self, 
                 latent_dim=64,
                 num_embeddings=512,
                 num_residual_blocks=2,
                 commitment_cost=0.25,
                 **kwargs):
        """
        Args:
            latent_dim: Dimension of latent space
            num_embeddings: Size of codebook
            num_residual_blocks: Number of residual blocks in encoder/decoder
            commitment_cost: Weight for commitment loss
        """
        super(VQVAE, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        # Build model components
        self.encoder = Encoder(latent_dim, num_residual_blocks)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(num_residual_blocks)
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        
    def call(self, inputs, training=None):
        """Forward pass through VQ-VAE."""
        encoded = self.encoder(inputs, training=training)
        quantized = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized, training=training)
        return reconstructed
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]
    
    def train_step(self, data):
        """Custom training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            encoded = self.encoder(data, training=True)
            quantized = self.vq_layer(encoded)
            reconstructed = self.decoder(quantized, training=True)
            
            # Reconstruction loss - MSE only
            reconstruction_loss = tf.reduce_mean((data - reconstructed) ** 2)
            
            # VQ losses from vq_layer
            vq_loss = tf.add_n(self.vq_layer.losses) if len(self.vq_layer.losses) > 0 else 0.0
            
            # Total loss
            total_loss = reconstruction_loss + vq_loss
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """Custom validation/test step."""
        # Forward pass
        encoded = self.encoder(data, training=False)
        quantized = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized, training=False)
        
        # Calculate losses
        reconstruction_loss = tf.reduce_mean((data - reconstructed) ** 2)
        
        # VQ losses from vq_layer
        vq_loss = tf.add_n(self.vq_layer.losses) if len(self.vq_layer.losses) > 0 else 0.0
        
        # Total loss
        total_loss = reconstruction_loss + vq_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }