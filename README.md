# SIS-Tween (Experimental)

This repository contains an **experimental deep learning pipeline for 2D video frame interpolation**. Given two input frames, the model predicts a single intermediate frame.

## Overview


Conceptually the architecture is like this:

<img width="758" height="378" alt="image" src="https://github.com/user-attachments/assets/1c9c8110-0e7d-4cff-999a-bf9a82da77cc" />


The pipeline is composed of three main components:

1. **Segmentator**
   Extracts segmentation masks from the two input frames.

2. **Interpolator**
   Interpolates the segmentation masks to estimate the intermediate structure.

3. **Synthesizer**
   Transforms the interpolated segmentation into a full RGB image.

> ⚠️ **Note:** The synthesizer is **not implemented yet**. At the moment, the pipeline only generates the interpolated segmented masks.

## Pipeline Flow

```
Input Frame 0 ──┐
                ├─▶ Segmentator ──▶ Interpolator ──▶ Synthesizer ──▶ Interpolated Frame
Input Frame 1 ──┘                                   (not implemented)
```

## Distributed Training

The project supports **Distributed Data Parallel (DDP)** training via PyTorch.

### Training

Run training with:

```bash
torchrun --nproc_per_node={number_gpus} train.py \
  --dataset_path {path}
```

You can adjust `--nproc_per_node` depending on the number of available GPUs.

### Testing

Run evaluation/testing with:

```bash
torchrun --nproc_per_node={number_gpus} test.py \
  --dataset_path {path}
```

## Requirements

* Python 3.8+
* PyTorch (with CUDA support recommended)
* Additional dependencies listed in `requirements.txt`

## Status

* Segmentator: ✅ Implemented
* Interpolator: ✅ Implemented
* Synthesizer: ❌ Not implemented
