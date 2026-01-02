# Image Super-Resolution using SRCNN and ESPCN

This project implements and compares two foundational deep learning architectures for **Image Super-Resolution (SR)**: the **Super-Resolution Convolutional Neural Network (SRCNN)** and the **Efficient Sub-Pixel Convolutional Network (ESPCN)**. The primary objective is to reconstruct high-resolution (HR) images from low-resolution (LR) inputs using the DIV2K dataset.

## Project Overview
Image Super-Resolution (SR) is the process of enhancing the quality and detail of a low-resolution image. This project explores two distinct approaches:

* **SRCNN:** A refinement-based approach. It first upscales the LR image using bicubic interpolation and then passes it through a three-layer CNN to map it to an HR output.
* **ESPCN:** An efficient approach that learns upsampling *within* the network. It uses **Pixel Shuffle** (sub-pixel convolution) layers to perform upscaling at the end of the network, significantly reducing computational costs by processing the image in LR space.



## Project Structure
* `Image Super-Resolution Using SRCNN.ipynb`: Full implementation of the SRCNN model, training pipeline, and evaluation.
* `Image_Super_Resolution_Using_ESPCN.ipynb`: Full implementation of the ESPCN model using sub-pixel convolution.
* `CVIP_Milestone2.pdf`: Detailed technical report on the project progress and comparative analysis.

## 📊 Dataset & Preprocessing
* **Dataset:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training, 100 validation, and 100 test images).
* **Preprocessing:** * Images are cropped to $128 \times 128$ pixels for training.
    * LR images are generated via bicubic downscaling with a scale factor of 2.
    * **SRCNN specific:** LR images are pre-upscaled back to $128 \times 128$ using bicubic interpolation before entering the model.
    * **ESPCN specific:** The model takes the raw $64 \times 64$ LR image as input.

## Architecture Details

### SRCNN
- **Input:** Pre-upscaled $128 \times 128$ image.
- **Layers:** Conv1 ($9 \times 9$) $\rightarrow$ Conv2 ($5 \times 5$) $\rightarrow$ Conv3 ($5 \times 5$).
- **Activation:** ReLU.

### ESPCN
- **Input:** Raw $64 \times 64$ LR image.
- **Layers:** Two convolutional layers ($5 \times 5$, $3 \times 3$) followed by a **Pixel Shuffle** layer.
- **Activation:** Tanh.



## Results
The models were evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

| Model | Avg PSNR | Avg SSIM | Avg MAE |
| :--- | :---: | :---: | :---: |
| **SRCNN** | 21.84 | 0.7123 | 0.0811 |
| **ESPCN** | **23.75** | **0.7546** | **0.0652** |

### Key Observations
1.  **ESPCN** consistently outperforms SRCNN in both quantitative metrics and visual sharpness.
2.  **SRCNN** results tend to be smoother (blurrier) because the model starts with a bicubic-upscaled image which already lacks high-frequency details.
3.  **ESPCN** is more computationally efficient because most of the feature extraction happens in the low-resolution space.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Pillow (PIL)
- Scikit-image (for PSNR and SSIM metrics)
