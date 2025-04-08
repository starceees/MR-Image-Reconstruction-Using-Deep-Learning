# MR Image Reconstruction Using Deep Learning

![GitHub last commit](https://img.shields.io/github/last-commit/starceees/MR-Image-Reconstruction-Using-Deep-Learning)
![License](https://img.shields.io/github/license/starceees/MR-Image-Reconstruction-Using-Deep-Learning)
![Stars](https://img.shields.io/github/stars/starceees/MR-Image-Reconstruction-Using-Deep-Learning?style=social)

A modular pipeline for 2D slice-wise cardiac MRI segmentation, comparing a custom UNet2D implementation against the self-configuring nnUNet framework on the Medical Segmentation Decathlon (MSD) Task02_Heart dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Rationale](#project-rationale)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualization](#results--visualization)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository provides an end-to-end pipeline for cardiac MRI segmentation using deep learning techniques. We compare our custom UNet2D implementation with the state-of-the-art nnUNet framework, analyzing performance differences, computational requirements, and implementation complexity.

## 🎯 Project Rationale

High-quality segmentation of cardiac MRI volumes is critical for clinical diagnosis and treatment planning. While nnUNet has set a new benchmark by automatically adapting to any medical segmentation task, there is still value in understanding and optimizing a hand-crafted UNet2D architecture under limited compute environments. This repository demonstrates both approaches, highlights their trade-offs, and provides a reusable, end-to-end codebase for future research.

## ✨ Features

- **Input Handling**: Accepts complex-valued MRI images by considering real and imaginary parts as separate channels.
- **High Capacity Models**: With up to 125 million trainable parameters, our implementation is capable of learning intricate patterns for accurate reconstruction.
- **Efficient Gradient Flow**: Carefully designed skip connections and up-convolutions ensure efficient backpropagation of gradients.
- **Customizability**: Flexible architecture supporting various configuration options for experimentation.
- **Comprehensive Evaluation**: Automated quantitative analysis using multiple metrics (Dice, Jaccard, precision, recall).
- **Visual Reporting**: Generates PDF reports with visualizations for qualitative assessment.

## 🏗️ Pipeline Architecture

### 1. Data Ingestion & Preparation
- **Download** the Task02_Heart dataset from the [MSD website](http://medicaldecathlon.com/) (not included in this repo).
- **Organize** as:
  ```
  Task02_Heart/
    ├── imagesTr/
    ├── labelsTr/
    └── imagesTs/
  ```
- **Configure** `data_root` in `parameters.yaml`.

### 2. Preprocessing & Augmentation
- **Axial Slice Extraction** from 3D NIfTI volumes.
- **Intensity Normalization** (min–max scaling).
- **On-the-fly Augmentations**: random flips, rotations, zoom.

### 3. Model Definitions
- **Custom UNet2D**: Fixed encoder–decoder with skip-connections, mixed-precision training (FP16), Cross-Entropy loss.
- **nnUNet**: Self-configuring pipeline that selects 2D/3D architectures, patch sizes, hyperparameters, and advanced augmentations automatically.

### 4. Training & Validation
- **Training Script** (`Implementation/training.py`):
  - Adam optimizer + ReduceLROnPlateau scheduler
  - WandB logging for real-time metrics
  - Checkpointing best validation Dice
- **Inference Script** (`Implementation/inference.py`):
  - Loads best model
  - Computes Dice, Jaccard, precision, recall on test set
  - Generates PDF visualizations

## 📊 Key Findings

| Metric            | Custom UNet2D | nnUNet | Δ      |
|-------------------|---------------|--------|--------|
| Average Dice      | 0.387         | 0.932  | +0.545 |
| Median Dice       | 0.439         | 0.933  | +0.494 |

- **nnUNet** delivers substantially higher segmentation accuracy and more precise mask alignment, thanks to its adaptive architecture and advanced data handling.
- **Custom UNet2D** remains a viable lightweight solution but requires extensive manual tuning and lacks inter-slice context.

## 📁 Repository Structure

```
MR-Image-Reconstruction-Using-Deep-Learning/
├── Data/                          # Data processing utilities
├── Implementation/                # Core implementation files
├── deprecated/                    # Legacy code (maintained for reference)
├── home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/ # Model results
├── interface/                     # User interface components
├── nnUNet/                        # nnUNet integration
├── .gitignore                     # Git ignore file
├── LICENSE                        # Project license
├── LICENSE.md                     # License details
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── resume_robotics.pdf            # Additional documentation
└── task_HSS.pdf                   # Task definition
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/starceees/MR-Image-Reconstruction-Using-Deep-Learning.git
   cd MR-Image-Reconstruction-Using-Deep-Learning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the data**:
   - Download the Task02_Heart dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
   - Place it in the expected directory structure
   - Update the configuration in `parameters.yaml`

## 💻 Usage

### Training the Custom UNet2D Model

```bash
python Implementation/training.py
```

### Running Inference and Evaluation

```bash
python Implementation/inference.py
```

### Running the nnUNet Benchmark

Follow nnUNet's official instructions to train on Task02_Heart. Place results under `nnUNet/nnunet_inference/` for direct comparison.

## 🖼️ Results & Visualization

Our implementation automatically generates comprehensive PDF reports with segmentation visualizations and quantitative metrics. Example visualizations are included in the `home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/` directory.

![Sample Segmentation](https://via.placeholder.com/800x400?text=Sample+Segmentation+Visualization)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms of the LICENSE file included in the repository. See [LICENSE](LICENSE) for more details.

---
