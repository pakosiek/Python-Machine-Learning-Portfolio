# Deep Learning & Medical Diagnosis Benchmarks

A comprehensive collection of Machine Learning and Deep Learning implementations, focusing on **binary classification problems** in medical diagnostics and non-linear data distributions.

The project demonstrates the transition from classical linear models to Deep Neural Networks (MLP), culminating in the analysis of real-world datasets from the UCI Machine Learning Repository.

## Project Overview

This repository contains a unified Jupyter Notebook that covers the following modules:

### 1. Theoretical Foundations & Limitations
* **Linear Separability:** Testing single-neuron perceptrons on synthetic data.
* **Non-Linearity:** Implementing Multi-Layer Perceptrons (MLP) to solve complex classification problems (XOR-like patterns).
* **Decision Boundary Visualization:** Plotting probability regions to visualize model behavior.

### 2. Engineering & Optimization
* **Hyperparameter Tuning:** Systematic analysis of Learning Rate, Batch Size, and Epochs.
* **Optimizer Comparison:** Benchmarking **SGD** vs. **Adam**.
* **Activation Functions:** Impact of **ReLU**, **Tanh**, and **Sigmoid** on convergence speed.

### 3. Medical Diagnostics (Real-World Application)
* **Diabetes Detection (Pima Indians Dataset):** * Data preprocessing using `StandardScaler`.
    * Evaluation using medical metrics: **Precision** and **Recall** (minimizing false negatives).
* **Breast Cancer Classification (UCI Repository):**
    * Comparing shallow vs. deep architectures for tumor classification (Malignant vs. Benign).
