# Speech Decoding Neuroprosthesis

This repository contains a suite of deep learning models and training pipelines designed for decoding speech directly from neural signals (e.g., ECoG/EEG). The project leverages state-of-the-art architectures like **Conformers**, **Vision Transformers (ViT)**, and **GRU-based decoders** to translate high-dimensional neural activity into phonetic sequences.

## Overview

The goal of this project is to develop a robust neuroprosthesis for speech restoration. By processing neural features extracted from brain-computer interfaces (BCI), the system learns to predict phonemes using Connectionist Temporal Classification (CTC) loss.

### Key Features
- **Diverse Model Architectures**:
  - **Conformers**: Combining Transformers and Convolutions for sequential neural data.
  - **Recurrent Decoders**: Multi-layer GRU and LSTM models with skip connections and optional strided patching.
  - **Vision Transformers (ViT)**: Treating neural spectral features as image patches.
  - **Hybrid CNN-Transformers**: CNN-based feature extraction followed by Transformer layers for global context.
- **Day-Specific Alignment**: Advanced logic to handle inter-day neural variability through learnable day-specific projection layers.
- **Robust Data Loading**: Advanced HDF5 dataloaders that handle variable-length trials, feature shape mismatches (e.g., transposing 512-channel data), and inconsistent field naming.
- **Training Pipeline**: Integrated training scripts with support for:
  - **Mixed Precision**: BF16 and FP16 support.
  - **Memory Efficiency**: Gradient checkpointing and 8-bit AdamW.
  - **Performance**: Support for `torch.compile` and Scaled Dot Product Attention (SDPA).
- **Phonetic Decoding**: Connectionist Temporal Classification (CTC) for mapping continuous neural signals to discrete phonemes.

---

## Project Structure

```text
.
├── Version1/               # Initial training scripts and experiments
├── Version2/               # Cleaned, robust training skeleton (CTC + Conformer)
│   ├── speech_decoding/    # Main package for model components
│   ├── train.py            # Primary training script
│   └── requirements.txt    # Version-specific dependencies
├── utils/                  # Core utility functions (Data loaders, Transforms)
│   ├── dataloader.py       # HDF5 Trial loaders
│   └── transforms.py       # Neural data augmentation and processing
├── notebooks/              # Exploratory notebooks and Vast.ai run logs
│   ├── Vit_V1.ipynb
│   ├── ViT_V2.ipynb
│   └── DirectSpectralDecoding-VastRun.ipynb
├── data/                   # Data summaries and metadata
└── Reference-Papers/       # Scientific literature supporting the implementation
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Speech-Decoding-Neuroprosthesis.git
   cd Speech-Decoding-Neuroprosthesis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r Version2/requirements.txt
   ```
   *(Note: Ensure you have PyTorch 2.0+ installed for `torch.compile` and SDPA support.)*

---

## Training Guide

The most robust training pipeline is located in `Version2/`. 

### Quick Start
To start training on a dataset of HDF5 files:

```bash
python Version2/train.py \
    --data-root /path/to/your/hdf5/data \
    --out ./experiments/conformer_run1 \
    --batch-size 8 \
    --epochs 50 \
    --amp bf16
```

---

## Evaluation
The models are evaluated using a **Phoneme Error Rate (PER)** proxy, calculated via the edit distance between greedy decoded phoneme sequences and ground truth labels.

---

## Model Architectures
- **Conformer**: Combines Transformers (global context) with Convolutions (local context) for superior sequence modeling of neural time-series.
- **GRU/LSTM Decoders**: Robust recurrent baselines with learnable initial states, day-specific projection headers, and support for strided temporal patching.
- **CNN-Transformer Hybrid**: Uses a 1D-CNN feature extractor (GELU activations, BatchNorm) to reduce temporal resolution before feeding into a multi-head Transformer encoder.
- **Vision Transformer (ViT)**: Reinterprets spectral neural features as a sequence of patches, similar to image processing.
- **Day-Specific Alignment**: To address "channel shift" and physiological changes across recording days, models include a shared backbone and unique, learnable linear layers for each recording session.
- **RoPE & Positional Encoding**: Implementation includes both Rotary Positional Embeddings (RoPE) for Conformers and standard Sinusoidal/Learned embeddings for Transformers.
