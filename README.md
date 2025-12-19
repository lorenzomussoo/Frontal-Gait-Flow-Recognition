# Frontal Gait-Flow Recognition
**Advanced Multimodal Biometric Identification System using Optical Flow and IMU Kinematics.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements a state-of-the-art forensic biometric system that identifies individuals by their unique gait (walking) patterns. By fusing **Visual Motion Data** (RGB-Depth Optical Flow) with **Kinematic Data** (Wearable IMU Sensors), the system maintains high precision across various terrains, including flat ground, stairs, and slopes.

---

## Technical Methodology

### 1. Visual Descriptor (GOFI)
The system extracts temporal dynamics using a **Global Optical Flow Image (GOFI)**. It uses the Farneback algorithm to compute dense motion vectors, which are then aggregated over a gait cycle to create a spatial representation of movement energy.

### 2. Kinematic Features
The system processes 19 data streams from IMU sensors (Acceleration, Gyroscope, Magnetometer, and Joint Angles), extracting **95 statistical features** (Mean, Std, RMS, Range, etc.) per segment.

### 3. Classification
Data is fused and passed through a **PCA-SVM** pipeline:
- **PCA**: Dimensionality reduction (99% variance).
- **SVM (RBF Kernel)**: High-dimensional classification for subject identification.

---

## Repository Structure

### Root
*   `Report.pdf`: Comprehensive technical report detailing the study, methodology, and results.
*   `analysis/`: Contains performance evaluation plots.
    *   `confusion_matrix_*.png`: Visualizes classification accuracy per subject.
    *   `confidence_analysis_*.png`: Reliability and error distribution analysis.
*   `model_multimodal/`: Pre-trained models and configurations.
    *   `svm_*.joblib`: Serialized SVM models.
    *   `best_params_*.json`: Hyperparameters found during Grid Search.

### code/ (Source Files)
#### **Core Pipeline**
*   `gait_processing.py`: Core Computer Vision logic (Background subtraction, Optical Flow).
*   `multimodal_feature_extractor.py`: Fuses Video and IMU data into unified feature vectors.
*   `train_walk_stairs.py` & `train_slope.py`: Training scripts for different terrain models.

#### **Demos & Visualization**
*   `predict_visual.py`: Forensic tool showing "Gait DNA" and movement analysis.
*   `predict_demo_security.py`: High-tech simulation of a biometric security checkpoint.
*   `visualize_result.py`: General result plotting utility.

#### **Data Integrity & Audit (Forensic Focus)**
*   `check_duplicates.py` / `check_video_duplicates.py`: MD5 hashing to prevent training/test leakage.
*   `fix_and_audit.py` / `audit_video_health.py`: Dataset cleaning and synchronization verification.
*   `verify_dataset_completeness.py`: Ensures all subjects have matching IMU and Video files.

#### **Utility & Maintenance**
*   `organize_imu.py` / `cleanup_imu.py`: Pre-processing and cleaning of raw sensor CSVs.
*   `convert_bags.py` / `explore_bag.py`: Tools for handling ROS bag files (dataset source).
*   `ablation_study.py`: Evaluates the impact of removing specific features or sensors.

---

## Getting Started

### Installation
Ensure you are using Python 3.12+.
```bash
git clone https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git
cd Frontal-Gait-Flow-Recognition
pip install numpy opencv-python pandas scikit-learn joblib rich
