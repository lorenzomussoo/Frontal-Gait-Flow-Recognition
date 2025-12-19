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

The system implements a **Multimodal Gait Recognition** framework designed to handle heterogeneous data sources (RGB-D Video and Inertial Sensors).

### 1. Visual Descriptors (Video)
*Applicable to **Walk** and **Stairs** tasks.*
To capture temporal motion patterns, the vision pipeline employs two complementary optical flow techniques:
* **GOFI (Gait Optical Flow Image):** Utilising **Dense Optical Flow** (Farneback algorithm), we accumulate the magnitude of flow vectors over the gait cycle to generate a spatial energy map.
* **Trace Map:** Utilising **Sparse Optical Flow** (Lucas-Kanade with Shi-Tomasi detection), we track specific anatomical key-points to create a skeletal history of limb trajectories.
* **Preprocessing:** Background is removed using a temporal median filter ($N=20$) and Otsu's binarisation.

### 2. Kinematic Features (IMU)
*Applicable to **All Tasks** (Sole modality for **Slope**).*
Since raw inertial logs vary in length based on the walking speed, we apply **Statistical Feature Extraction** to the 19 raw CSV logs (Xsens).
For each sensor channel (Acceleration, Gyroscope, Joint Angles), we compute a **5-dimensional descriptor**:
* Mean ($\mu$)
* Standard Deviation ($\sigma$)
* Minimum ($min$)
* Maximum ($max$)
* Root Mean Square ($RMS$)

> **Note:** The **Slope** dataset contains exclusively inertial data. Consequently, the system utilizes a specialized IMU-Only pipeline for these tasks, achieving 100% accuracy using kinematic features alone.

### 3. Classification Architecture
The extracted features are processed through a **two-branch learning architecture** tailored to the specific modality availability:

* **Feature Spaces & Dimensionality Reduction (PCA):**
    We apply Principal Component Analysis (retaining **95% variance**) to compress the input space while filtering noise:
    * **Main Model (Walk/Stairs):** The high-dimensional multimodal vector is reduced from **54,082 $\to$ 353 principal components**.
    * **Slope Model (IMU Only):** The inertial feature vector is reduced from **4,930 $\to$ 53 principal components**.
* **Classifier:** Both branches utilize a Support Vector Machine (**SVM**) with a **Linear Kernel** ($C=1$). The Linear kernel was selected after Grid Search proved that both the massive multimodal space and the compact inertial space are linearly separable.

---

## Repository Structure

### Root
* `Report.pdf`: Comprehensive technical report detailing the study, methodology, and results.
* `Presentation.pdf`
* `analysis/`: Contains performance evaluation plots.
    * `confusion_matrix_*.png`: Visualises classification accuracy per subject (Walk/Stairs and Slope).
    * `confidence_analysis_*.png`: Reliability and error distribution analysis with probability density curves.
* `model_multimodal/`: Pre-trained models and configurations.
    * `multimodal_svm_walk_stairs.joblib`: Main SVM model (Video + IMU).
    * `svm_slope.joblib`: Slope-specific SVM model (IMU Only).
    * `best_params_*.json`: Optimal hyperparameters ($C$, $Gamma$, PCA components) found during Grid Search.

### Source Code (`code/`)

#### **1. Core Machine Learning Pipeline**
* `gait_processing.py`: Core Computer Vision logic. Implements Background Subtraction (Median Filter), Dense Optical Flow (GOFI), and Sparse Optical Flow (Lucas-Kanade).
* `multimodal_feature_extractor.py`: Fuses Video and IMU data into unified feature vectors for training.
* `train_walk_stairs.py`: Trains the Main Multimodal Model (PCA + Linear SVM).
* `train_slope.py`: Trains the specialized Slope Model (IMU-Only).

#### **2. Data Engineering & ETL (Extract, Transform, Load)**
Scripts handling the transition from raw ROS bags and CSVs to the canonical dataset.
* `convert_bags.py`: Parses binary ROS bags, decodes RGB-D streams, and normalizes depth.
* `organize_imu.py`: Manages inertial data ingestion, resolving subject naming conflicts and merging sessions.
* `explore_bag.py`: Utility to inspect the structure of raw ROS bag files.
* `cleanup_imu.py`: Pre-processing and cleaning of raw sensor CSVs.
* `fix_targets.py`: Utility for correcting label inconsistencies in the dataset.

#### **3. Forensic Auditing & Data Integrity (QA)**
Strict validation scripts to ensure zero data leakage and dataset health.
* `check_duplicates.py` / `check_video_duplicates.py`: Calculates MD5/SHA256 hashes to verify physical separation between Train and Test sets.
* `check_basename.py`: Validates the logical consistency of the Train/Test split (**Walk: Runs 5-6 Test; Others: Run 3 Test**) to ensure zero data leakage.
* `check_dims.py`: Verifies the structural integrity and dimensionality of processed feature vectors.
* `audit_video_health.py`: Flags corrupted video files or samples with insufficient frame counts.
* `verify_dataset_completeness.py`: Ensures 100% alignment between IMU and Video files for all subjects.
* `fix_and_audit.py`: Automated repair script for common dataset inconsistencies.
* `cleanup_processed.py`: Removes temporary or intermediate files from the processing pipeline.

#### **4. Analysis, Demos & Visualisation**
* `predict_demo_security.py`: **"Gait Security Pro"**. A real-time simulation of a biometric access control checkpoint with AR visualisation.
* `predict_visual.py`: **"Gait Visualizer"**. An XAI (Explainable AI) forensic tool showing "Gait DNA" matrices.
* `analyze_confidence.py`: Calibrates model probabilities to determine the optimal security rejection threshold.
* `ablation_study.py`: Performs comparative experiments (Video vs. IMU vs. Fusion) to quantify modality impact.
* `visualize_result.py`: Generates confusion matrices and performance plots.

---

## Getting Started

### Installation
Ensure you are using Python 3.12+.
```bash
git clone [https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git](https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git)
cd Frontal-Gait-Flow-Recognition
pip install numpy opencv-python pandas scikit-learn joblib rich
