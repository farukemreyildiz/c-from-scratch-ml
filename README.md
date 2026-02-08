# C From-Scratch Machine Learning

A low-level image classification project implemented entirely in C without any machine learning libraries.

This project builds a complete training pipeline from scratch including:
- image loading
- dataset creation
- linear model
- forward pass
- backpropagation
- multiple optimizers
- performance tracking

The goal is to understand how machine learning algorithms work internally rather than relying on high-level frameworks.

---

## Features

### Dataset Handling
- PGM image loader
- Automatic directory scanning
- Dynamic dataset creation
- Train/Test split
- Shuffle support

### Model
- Linear classifier
- Tanh activation
- Manual gradient computation

### Optimizers
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Adam Optimizer

### Training Utilities
- Loss tracking
- Accuracy tracking
- Epoch timing
- Result logging to files

---

## Technologies

- C (no ML libraries)
- stb_image for image reading
- Standard C math & file handling

---

## Project Structure

