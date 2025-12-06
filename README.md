# ⚙️ MLOPs Assignment Option A: Optimizing the Development & Training Pipeline

## 📌 Assignment Overview

This repository contains an optimized machine learning pipeline for classifying rock, paper, and scissors hand gestures from images. The pipeline automates **environment setup**, **data preprocessing**, and **model training**, all orchestrated through a single entry point: `run_entire_dev_pipeline.py`.

Originally limited to CPU execution with slow preprocessing and suboptimal model performance, this assignment focuses on **GPU acceleration**, **batch optimization**, and **hyperparameter tuning** to significantly improve runtime and model accuracy.

YOU will FORK this Repo, work on the fork, and when complete, request a PR to complete the assignment.

---

## 🎯 Objectives

To meet the optimization goals, this project improves on the baseline pipeline by:

- ✅ Reducing **total runtime** to under 1 hour  
- ✅ Achieving a **model accuracy of 97% or higher**  
- ✅ Ensuring full **GPU (CUDA 11.8)** utilization for preprocessing and training  
- ✅ Removing bottlenecks in image preprocessing and training  
- ✅ Maintaining a **modular, readable, and automated** codebase  

---
## 📂 Project Structure

├── run_entire_dev_pipeline.py # 🚀 Main orchestrator script
├── create_virtual_env.py # 🔧 Virtual environment + dependency setup
├── setup_training.py # ⚙️ Data setup and preprocessing entry point
├── train_model.py # 🧠 Model training script
├── datautils.py # 🖼️ Image preprocessing (resizing, grayscale, background removal)
├── model_utils.py # 🤖 Model architecture, training logic, and ONNX export
├── requirements.txt # 📦 Dependency list (GPU-compatible versions)
├── Image_Dataset.zip # 📁 Compressed dataset (automatically extracted)
├── output/ # 📤 Trained model export folder
└── README.md

---

## 🔍 Pipeline Breakdown

### 📜 `run_entire_dev_pipeline.py`
Main entry point for the entire workflow. It:
- Creates the virtual environment
- Installs dependencies (GPU-compatible)
- Runs dataset extraction and preprocessing
- Triggers model training and export

### 📜 `create_virtual_env.py`
- Creates a Python virtual environment dynamically  
- Installs:
  - `torch` with CUDA 11.8 support  
  - `onnxruntime-gpu`  
  - `rembg` for background removal  
- Automatically extracts `Image_Dataset.zip`  

### 📜 `setup_training.py` ➝ `datautils.py`
- Preprocesses all input images:
  - Resize  
  - Grayscale conversion  
  - **GPU-accelerated background removal** via ONNX  
- Supports **batch processing** for improved performance  

### 📜 `train_model.py` ➝ `model_utils.py`
- Model definition and training loop  
- Uses `torch.device("cuda")` when GPU is available  
- Improved configuration:
  - 15+ epochs  
  - Better learning rate  
  - Enhanced augmentation  
- Exports final model to `output/` in ONNX format  

---

## ⚡ Optimizations Applied

| Area             | Before                       | After                            |
|------------------|------------------------------|----------------------------------|
| Preprocessing     | 15–20 min (CPU)              | 5–8 min (GPU, batch-optimized)   |
| Training          | 5 epochs (CPU, underfit)     | 15+ epochs (GPU-accelerated)     |
| Model Accuracy    | ~70%                         | ≥ 97%                            |
| Total Runtime     | > 1h 15m                     | ⏱️ < 1 hour                      |
| Automation        | CPU-only                     | Fully GPU-enabled automation     |

---

## 🚀 How to Run

### 1. Clone the Repository

git clone https://github.com/your-username/MLOPs_Assignment_Option_A.git
cd MLOPs_Assignment_Option_A

python run_entire_dev_pipeline.py

💡 This script will:

Create a new virtual environment

Install GPU-compatible dependencies

Extract and preprocess the dataset

Train the model using GPU

Export the final model to /output/model.onnx

### 2. 🧪 Requirements

OS: Linux / Windows with WSL2

GPU: NVIDIA with CUDA 11.8

Python: 3.8–3.10

Disk space: ~2GB

NVIDIA drivers and nvidia-smi must be installed

### 3. 📝 Assessment Criteria
Metric	Target	Weight
Preprocessing Time	Under 20 minutes	50%
Training Time	Under 20 minutes	35%
Model Accuracy	≥ 97%	15%
✅ Deliverables

🔁 Fully automated pipeline script: run_entire_dev_pipeline.py

⚡ Optimized preprocessing using GPU and ONNX

🎯 High-accuracy model exported to ONNX format

📂 output/ directory containing final model

🧩 Modular Python scripts for maintainability

📘 Documentation (this README)