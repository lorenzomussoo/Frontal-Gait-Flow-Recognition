# Frontal Gait-Flow Recognition

**Multimodal Biometric Identification System using Optical Flow and IMU Kinematics.**

This project implements a robust forensic biometric system capable of identifying individuals based on their unique gait patterns. By fusing **Visual Data** (RGB-Depth) and **Kinematic Data** (IMU Sensors), the system achieves state-of-the-art accuracy across diverse scenarios, including flat ground walking, stair climbing, and slope navigation.

---

## Key Features

*   **Multimodal Fusion**: Combines dense optical flow energy images (GOFI) with 95 statistical features from wearable IMU sensors.
*   **Robust Preprocessing**:
    *   Static Background Subtraction ($N=20$).
    *   Farneback Dense Optical Flow (Hue-encoded directionality).
    *   Lucas-Kanade Sparse Feature Tracking (Visualizing limb swing).
*   **Data Integrity**: Built-in anti-leakage verification system using MD5 hashing to ensure mathematically disjoint training and test sets.
*   **Forensic Visualization**:
    *   **Gait DNA Matrix**: Explains predictions by comparing subject features with reference clusters.
    *   **Security Pro Terminal**: Simulates a real-world access control system with "Heatmap", "Cyber Edges", and "Vibrant Depth" visual filters.

---

## Project Structure

```bash
Gait-Flow-Recognition/
├── analysis/                # Statistical analysis and plots
├── code/                    # Source code
│   ├── gait_processing.py            # Core visual algorithms (Background, Optical Flow, LK)
│   ├── multimodal_feature_extractor.py # Feature extraction pipeline (Video + IMU)
│   ├── train_slope.py                # SVM training for Slope tasks (IMU only)
│   ├── train_walk_stairs.py          # SVM training for Walk/Stairs (Multimodal)
│   ├── predict_visual.py             # Forensic Gait Visualizer Demo
│   ├── predict_demo_security.py      # Security Access Control Simulation
│   ├── check_duplicates.py           # Data leakage verification
│   └── ...                           # Utility scripts (cleaning, auditing)
├── model_multimodal/        # Saved SVM models (.joblib) and PCA parameters
├── processed_features/      # Extracted .npy feature vectors (organized by Subject/Action)
└── project_report.tex       # Comprehensive LaTeX technical report
```

---

## Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies:
```bash
pip install numpy opencv-python pandas scikit-learn rich joblib
```

### 2. Feature Extraction
To process raw video headers and IMU CSVs into feature vectors:
```bash
python code/multimodal_feature_extractor.py
```
*   **Input**: Raw dataset in `/Volumes/LaCie/GAIT/dataset` (Adjust path in script).
*   **Output**: Normalized `.npy` vectors in `processed_features/`.

### 3. Data Verification
**Crucial Step**: Before training, verify that no test data has leaked into the training set.
```bash
python code/check_duplicates.py
```
*   **Green**: Test Passed.
*   **Red**: Leakage Detected (Do not proceed).

### 4. Training Models
Train the Support Vector Machines (SVM) with RBF Kernels:
```bash
# Main Model (Walk + Stairs) -> Video + IMU Fusion
python code/train_walk_stairs.py

# Slope Model (Slope Up/Down) -> IMU Only (Specialist Model)
python code/train_slope.py
```

### 5. Running Demos

#### Gait Visualizer (Forensic Mode)
Detailed analysis of specific run, showing the "Gait DNA" comparison matrix.
```bash
python code/predict_visual.py
```
*   **Controls**: `SPACE` to pause, `Q` to quit.

#### \U0001F6E1\uFE0F Security Pro System (Simulation)
Simulates a gate access terminal with advanced visual overlays.
```bash
python code/predict_demo_security.py
```
*   **Features**: Displays real-time confidence, status codes (Granted/Denied), and visual filters (Heatmap, Cyber Edges).

---

## Methodology Highlight

The system relies on a **Global Optical Flow Image (GOFI)**, which aggregates temporal motion into a single spatial descriptor:

$$ \text{GOFI}(x,y) = \sum_{t=1}^{T} \vec{V}_t(x,y) $$

This is concatenated with a **Statistical IMU Vector** $\mathbf{v}_{imu} \in \mathbb{R}^{95}$, comprised of Mean, Std, Min, Max, and RMS for 19 sensor streams (Acceleration, Gyro, Magnetometer, Joint Angles).

For classification, we use **PCA (0.99 variance)** followed by an **SVM (RBF Kernel)**, achieving >95% accuracy on the multimodal test set.

---

## Author

**Lorenzo Musso and Giulia Pietrangeli**
