# MR-Image-Reconstruction-Using-Deep-Learning
This repository contains a modified U-Net architecture implementation specifically tailored for the reconstruction of MRI images. The U-Net model is a powerful convolutional neural network known for its efficiency in image segmentation tasks, and in this case, has been adapted to handle the complex data typically associated with MRI scans.

# Model Overview
The U-Net model in this repository features a contracting path to capture context and an expansive path for precise localization. Modifications have been made to the original U-Net architecture to accommodate the real and imaginary parts of MRI data, with the following structure:

A series of DoubleConv blocks for convolution operations followed by batch normalization and ReLU activation.
An additional downsampling step to increase the model's depth for capturing finer details in the images.
Skip connections at each level of the expansive path to aid gradient flow and improve learning.

# Features
Input Handling: Accepts complex-valued MRI images by considering real and imaginary parts as separate channels.
High Capacity: With over 125 million trainable parameters, our model is capable of learning intricate patterns for accurate reconstruction.
Gradient Flow: Carefully designed skip connections and up-convolutions ensure efficient backpropagation of gradients.
Customizability: The model's architecture can be easily adapted for various input sizes and reconstruction targets.

# Usage
The repository includes the full PyTorch implementation of the model, along with scripts to demonstrate training and inference processes. Users can leverage the provided code to train the model on their datasets, perform MRI image reconstruction, and validate the results against ground-truth data.

# Dependencies
- Python 3.8+
- PyTorch
- Torchvision
- torchviz (for model visualization)

# Contributing
We welcome contributions to improve the model further. Please feel free to fork the repository, make your changes, and submit a pull request.

# License
Distributed under the MIT License. See LICENSE for more information.

# Contact
Your Name - @RaghuRamCS1 - raghuram2309@gmail.com

Project Link: https://github.com/starceees/MR-Image-Reconstruction-Using-Deep-Learning
