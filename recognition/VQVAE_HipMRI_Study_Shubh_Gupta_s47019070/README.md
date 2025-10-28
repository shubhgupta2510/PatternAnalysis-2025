# VQ-VAE Implementation for HipMRI 2D Slice Reconstruction

**Author:** Shubh Gupta (s47019070)

## Overview

This project implements a Vector Quantized Variational Autoencoder (VQ-VAE) for reconstructing 2D MRI slices from the CSIRO HipMRI dataset [[1](https://data.csiro.au/collection/csiro:51392v2?redirected=true)]. The primary objective is to develop a generative model capable of learning compressed representations of medical imaging data and reconstructing high-quality images with a Structural Similarity Index Measure (SSIM) exceeding 0.6.

### What is VQ-VAE?

Traditional Variational Autoencoders (VAEs) encode images into continuous latent spaces, which can lead to issues such as posterior collapse where the model ignores latent variables during reconstruction [[3](https://arxiv.org/abs/1711.00937)]. The VQ-VAE addresses this limitation by introducing discrete latent representations through vector quantization.

### How VQ-VAE Works

The VQ-VAE architecture consists of three main components:

1. **Encoder**: Compresses input images into continuous latent vectors
2. **Vector Quantization Layer**: Maps continuous encoder outputs to discrete codebook vectors by finding the nearest embedding in a learned codebook
3. **Decoder**: Reconstructs images from the quantized latent representations

This discrete bottleneck forces the model to learn more structured and meaningful representations compared to standard VAEs. The quantization process creates a fixed set of embedding vectors (codebook) that the model learns during training, allowing for better control over the latent space [[2](https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337)].

### Architecture Visualization

![VQ-VAE model architecture](resources/vqvae_diagram.webp)
*Figure: VQ-VAE architecture showing encoder, vector quantization, and decoder components* [[2](https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337)]

### Loss Function Components

The model is optimized using a composite loss function with three terms:

1. **Reconstruction Loss**: Quantifies pixel-wise differences between original and reconstructed images using mean squared error
2. **Codebook Loss**: Moves codebook embeddings closer to encoder outputs to improve quantization accuracy
3. **Commitment Loss**: Encourages encoder outputs to stay close to chosen embeddings, preventing the encoder from arbitrarily changing its output scale

![VQ-VAE loss function](resources/loss_function.webp)
*Figure: Mathematical formulation of the VQ-VAE loss function* [[2](https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337)]

The combined loss ensures that both the discrete codebook and continuous encoder/decoder networks are jointly optimized for high-quality reconstructions.

## Dataset Description and Preprocessing

The implementation uses the CSIRO HipMRI dataset [[1](https://data.csiro.au/collection/csiro:51392v2?redirected=true)], which contains 12,660 grayscale MRI slices of the pelvic region from male patients. The dataset preparation code is implemented in `dataset.py`.

### Dataset Characteristics

- **Total Images**: 12,660 2D MRI slices
- **Image Format**: Grayscale
- **Primary Dimensions**: Most images are 256×128 pixels
- **Anatomical Region**: Male pelvic region
- **File Format**: NIfTI (.nii.gz)

### Data Split Rationale

The dataset was partitioned into training, validation, and test sets using an approximate 90.5%/5.2%/4.3% split:

| Split      | Image Count | Percentage |
| ---------- | ----------- | ---------- |
| Training   | 11,460      | 90.5%      |
| Validation | 660         | 5.2%       |
| Test       | 540         | 4.3%       |
| **Total**  | **12,660**  | **100%**   |

This split ensures sufficient training data while reserving adequate samples for validation during training and final evaluation. The larger training set is necessary for the model to learn the diverse anatomical variations present in medical imaging. The validation set provides enough samples to monitor overfitting, while the test set offers a robust evaluation of generalization performance.

### Preprocessing Pipeline

All preprocessing is handled by the data loading pipeline with the following transformations:

1. **Resizing**: All images standardized to 256×128 pixels to ensure uniform input dimensions
2. **Normalization**: Pixel values normalized to zero mean and unit variance to stabilize training and help the model converge faster

No data augmentation was applied to preserve the anatomical accuracy of the medical images, as geometric transformations could introduce artifacts not representative of real clinical data.

## Model Architecture

The complete model architecture is defined in `modules.py`. The implementation consists of an encoder-decoder structure with vector quantization in the latent space.

<details>
<summary>VQ-VAE model structure (need to be CLICKED)</summary>
<br>
<pre>
VQVAE(
  (encoder): Encoder(
    (conv1): Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (residual_stack): ResidualStack(
      (stack): Sequential(
        (0): ResidualLayer(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (relu): ReLU()
        )
        (1): ResidualLayer(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (relu): ReLU()
        )
      )
    )
    (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  )
  (vector_quantization): VectorQuantizer(
    (embedding): Embedding(512, 64)
  )
  (decoder): Decoder(
    (conv1): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (residual_stack): ResidualStack(
      (stack): Sequential(
        (0): ResidualLayer(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (relu): ReLU()
        )
        (1): ResidualLayer(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
          (relu): ReLU()
        )
      )
    )
    (conv2): ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  )
)
</pre>
</details>

<details>
<summary>Residual Stack model structure (need to be CLICKED)</summary>
<br>
<pre>
ResidualStack(
  (stack): Sequential(
    (0): ResidualLayer(
      (conv1): Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
      (relu): ReLU()
    )
    (1): ResidualLayer(
      (conv1): Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
      (relu): ReLU()
    )
  )
)
</pre>
</details>

<details>
<summary>Residual Layer model structure (need to be CLICKED)</summary>
<br>
<pre>
ResidualLayer(
  (conv1): Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
  (relu): ReLU()
)
</pre>
</details>

## Training Methodology

The training implementation is contained in `train.py`. The model was trained end-to-end using the Adam optimizer with a multi-component loss function.

### Training Configuration

The following hyperparameters were selected through empirical experimentation:

| Parameter            | Value  | Rationale                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| Batch Size           | 16     | Balanced GPU memory utilization with stable gradient estimates                                          |
| Total Epochs         | 100    | Sufficient for convergence based on validation SSIM plateauing                                          |
| Learning Rate        | 1e-4   | Provided stable training without oscillations; tested against 1e-3 (unstable) and 1e-5 (too slow)       |
| Embedding Dimension  | 64     | Captured sufficient detail without overfitting; higher values showed diminishing returns                |
| Codebook Size        | 512    | Number of discrete embedding vectors in the quantization codebook                                       |
| Commitment Cost (β)  | 0.25   | Balanced encoder commitment to embeddings; standard value from VQ-VAE literature [[3](https://arxiv.org/abs/1711.00937)] |
| Hidden Channels      | 64     | Determined encoder/decoder capacity                                                                     |

### Loss Function Implementation

The training objective combines three loss terms:

**L_total = L_reconstruction + L_codebook + β × L_commitment**

Where:
- **L_reconstruction**: Mean Squared Error (MSE) between input and reconstructed images
- **L_codebook**: MSE between encoder output and nearest codebook embedding (updates codebook)
- **L_commitment**: MSE between encoder output and nearest codebook embedding (updates encoder)

The commitment cost β prevents the encoder from arbitrarily increasing the magnitude of its outputs.

### Training Process

During each training epoch:

1. Images are passed through the encoder to produce continuous latent vectors
2. Latent vectors are quantized by finding nearest codebook embeddings
3. Quantized vectors are decoded to reconstruct images
4. All three loss components are computed and combined
5. Gradients are backpropagated to update encoder, decoder, and codebook parameters
6. Validation SSIM is computed at the end of each epoch
7. Model checkpoint is saved when validation SSIM improves

The model was trained using the Adam optimizer which adapts learning rates for each parameter, helping to handle the different scales of the three loss components.

## Experimental Results

The model was evaluated on the held-out test set of 540 images after 100 training epochs. The final model achieved a mean SSIM of **0.789** on the test dataset, substantially exceeding the target threshold of 0.6.

### Training Dynamics

The figure below shows the progression of training loss, validation loss, and validation SSIM across epochs:

![Training Loss and SSIM](resources/loss_ssim_plot.png)

**Observations:**
- Training and validation losses decreased consistently, indicating effective learning without significant overfitting
- SSIM scores showed steady improvement before plateauing around epoch 70-80
- The close tracking of training and validation metrics suggests good generalization

## Qualitative Results: Reconstruction Examples

### Reconstruction Quality Over Training

The following images demonstrate the improvement in reconstruction quality throughout training:

**Early Training (Epoch 1)**
![Reconstructed Images at Epoch 1](resources/plots/epoch_1/epoch_1_images.png)

**Mid Training (Epoch 33)**
![Reconstructed Images at Epoch 33](resources/plots/epoch_33/epoch_33_images.png)

**Late Training (Epoch 70)**
![Reconstructed Images at Epoch 70](resources/plots/epoch_70/epoch_70_images.png)

**Final Model (Epoch 100)**
![Reconstructed Images at Epoch 100](resources/plots/epoch_100/epoch_100_images.png)

### Analysis of Reconstruction Quality

The visual results demonstrate that the model successfully captures the overall anatomical structure and intensity patterns of the pelvic MRI slices. However, some fine-grained details are smoothed out in the reconstructions. This is attributable to the quantized latent space having dimensions of 8×8, which results in each discrete embedding representing a 32×16 pixel region in the original image. While this compression ratio enables efficient representation learning, it inherently limits the preservation of high-frequency spatial details.

### Best and Worst Case Reconstructions

To understand the model's performance range, the images with highest and lowest SSIM scores on the test set are shown below:

**Top 4 Reconstructions by SSIM**
![Best Reconstructed Images](resources/reconstructed_images/best_reconstructed_images.png)

**Bottom 4 Reconstructions by SSIM**
![Worst Reconstructed Images](resources/reconstructed_images/worst_reconstructed_images.png)

The best reconstructions typically correspond to images with simpler anatomical patterns and higher contrast, while the worst cases involve more complex tissue boundaries and lower contrast regions where subtle details are more critical.

## Discussion: Limitations and Future Directions

### Current Limitations

1. **Spatial Resolution of Latent Space**: The 8×8 quantized latent space creates a significant compression bottleneck. Each discrete code represents a 32×16 pixel patch in the original image, which limits the model's ability to preserve fine anatomical details and sharp boundaries.

2. **Loss Function Scope**: The current implementation relies solely on MSE for reconstruction, which measures pixel-wise differences but doesn't capture perceptual similarity. This can lead to blurry reconstructions even when SSIM is relatively high.

3. **Limited Architectural Depth**: The current encoder and decoder use only two convolutional layers each. Deeper architectures might learn more hierarchical representations of the anatomical structures.

### Proposed Improvements

- **Increased Latent Resolution**: Implementing a 16×8 or 16×16 quantized latent space would allow each code to represent smaller image regions, potentially preserving more fine-grained details while still maintaining discrete representations.

- **Perceptual Loss Integration**: Incorporating perceptual loss functions (e.g., features from pre-trained networks) or adversarial training could improve the visual quality and sharpness of reconstructions.

- **Hierarchical VQ-VAE**: Using multiple levels of vector quantization at different scales could capture both coarse anatomical structure and fine details.

- **Extended Training**: While SSIM began plateauing around epoch 70-80, longer training with learning rate scheduling might yield marginal improvements.

## Reproducibility

### System Requirements

- **Python Version**: 3.10.15 or higher
- **GPU**: CUDA-compatible GPU recommended for training (training on CPU is prohibitively slow)

### Dependencies

Install the following Python packages:

| Package      | Version | Purpose                                      |
| ------------ | ------- | -------------------------------------------- |
| torch        | 2.5.1   | Deep learning framework                      |
| torchvision  | 0.20.1  | Image transformations and utilities          |
| torchaudio   | 2.5.1   | Audio processing (dependency)                |
| torchmetrics | 1.5.1   | SSIM and other evaluation metrics            |
| numpy        | 1.26.4  | Numerical operations                         |
| matplotlib   | 3.9.2   | Visualization and plotting                   |
| nibabel      | 5.3.2   | NIfTI medical image file I/O                 |
| tqdm         | 4.66.6  | Progress bars                                |

### Running the Code

**To train the model:**
```bash
python train.py
```
This will train the VQ-VAE model using the configuration specified in `config.py` and save model checkpoints based on validation SSIM.

**To evaluate on test data:**
```bash
python predict.py
```
This loads the best saved model and evaluates it on the test set, generating reconstruction visualizations and computing metrics.

**Note**: Ensure that the data paths in `dataset.py` point to the correct location of the CSIRO HipMRI dataset files.

## Conclusion

This project successfully implemented a VQ-VAE model for medical image reconstruction, achieving a test SSIM of 0.789, which significantly exceeds the target performance of 0.6. The model demonstrates strong capability in learning compressed discrete representations of pelvic MRI slices and reconstructing anatomically plausible images.

The training curves indicate stable learning dynamics with good generalization—validation metrics closely tracked training metrics without significant overfitting. The discrete latent space with 512 codebook embeddings proved sufficient for capturing the diversity of anatomical patterns in the dataset.

While the reconstructions successfully preserve overall anatomical structure and intensity distributions, the 8×8 latent resolution creates a bottleneck that limits fine detail preservation. The observed SSIM plateau around 0.78-0.80 suggests this represents an approximate performance ceiling for the current architecture. Further improvements would likely require architectural modifications such as increasing the latent resolution or incorporating hierarchical quantization, rather than simply extending training duration.

The project demonstrates the viability of discrete latent variable models for medical imaging applications and provides a foundation for future work exploring more sophisticated VQ-VAE variants or downstream tasks such as image generation from learned codebooks.

## References

1. CSIRO. (2020). *HipMRI Study Dataset*. CSIRO Data Access Portal. https://data.csiro.au/collection/csiro:51392v2?redirected=true

2. Patel, S. (2020). *An Overview on VQ-VAE: Learning Discrete Representation Space*. Analytics Vidhya. https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337

3. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. arXiv preprint arXiv:1711.00937. https://arxiv.org/abs/1711.00937

4. Papers With Code. *VQ-VAE Method*. https://paperswithcode.com/method/vq-vae