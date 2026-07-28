# Single Instance Object Detection 

> **Case Study**
>
> This project was developed as part of a technical case study on **Single Instance Object Detection**.
> It presents a practical pipeline for detecting, identifying and tracking a single target instance in real time.

---

# Overview
Instead of detecting every object in the scene, this project focuses on identifying and tracking one specific target instance using a reference image.

The proposed solution combines:

- **YOLO11** for object detection
- **OSNet** for person re-identification
- **Lightweight appearance matching** for vehicles
- **OSTrack** for visual tracking
- **Operator-assisted target reacquisition** after tracking failure

The implementation prioritizes real-time performance while remaining suitable for deployment on edge devices.

---

# Features

- Single Instance Object Detection
- Reference image based target selection
- YOLO11 object detection
- OSTrack visual tracking
- Person Re-Identification using OSNet
- Lightweight vehicle appearance matching
- Operator-assisted target reacquisition
- Modular C++ implementation
- Edge-oriented architecture

---

# System Pipeline

```text
Reference Image
        │
        ▼
YOLO11 Detection
        │
        ▼
Target Type Filtering
(Person / Vehicle)
        │
        ▼
Appearance Matching
┌─────────────────────────────────┐
│ Person  → OSNet Re-ID           │
│ Vehicle → Appearance Matching   │
└─────────────────────────────────┘
        │
        ▼
Best Candidate Selection
        │
        ▼
OSTrack Initialization
        │
        ▼
Target Tracking
        │
        ▼
Tracking Lost?
      /        \
    No          Yes
    │            │
    ▼            ▼
 Continue   Operator Defines ROI
                  │
                  ▼
        Target Re-Detection
                  │
                  ▼
      Reinitialize OSTrack
                  │
                  ▼
         Continue Tracking
```

---

# Workflow

## 1. Initialization

The application loads:

- YOLO11 object detector
- OSTrack tracker
- OSNet Person Re-ID model
- Reference target image

The reference image represents the specific object that should be identified inside the scene.

---

## 2. Object Detection

Each video frame is processed by YOLO11.

For every detected object, YOLO provides:

- Bounding box
- Class ID
- Detection confidence

---

## 3. Appearance Matching

The matching strategy depends on the target category.

### Person

Detected person crops are passed through **OSNet**.

OSNet extracts a feature embedding representing the person's appearance.

The cosine similarity between the reference embedding and each detected person is computed.

The candidate with the highest similarity score is selected.

### Vehicle

For vehicles, a lightweight appearance descriptor is used.

The implementation compares:

- Color distribution
- Aspect ratio
- YOLO confidence

A weighted score is computed for every detected vehicle.

The highest scoring candidate is selected.

---

## 4. Tracking

The selected candidate initializes **OSTrack**.

After initialization, OSTrack performs frame-by-frame tracking without repeatedly executing object detection.

---

## 5. Re-Detection

If the tracker confidence falls below a predefined threshold:

1. The operator draws an approximate ROI.
2. YOLO searches only inside this region.
3. Appearance matching is performed again.
4. OSTrack is reinitialized using the recovered target.

This approach avoids automatically switching to another object after temporary tracking failures.

---

# Project Structure

```text
SingleInstanceDetection/
│
├── build/
│
├── include/
│   ├── Detection.h
│   ├── ImageUtils.h
│   ├── OSTrackTracker.h
│   ├── PersonMatcher.h
│   ├── VehicleMatcher.h
│   ├── VideoPlayer.h
│   └── YOLODetector.h
│
├── src/
│   ├── main.cpp
│   ├── ImageUtils.cpp
│   ├── OSTrackTracker.cpp
│   ├── PersonMatcher.cpp
│   ├── VehicleMatcher.cpp
│   ├── VideoPlayer.cpp
│   └── YOLODetector.cpp
│
├── models/
│   ├── yolo11s.pt
│   ├── OSTrack_ep0300.pth.tar
│   └── osnet_ain_x0_5_imagenet.pth
│
├── models_onnx/
│   ├── yolo11s.onnx
│   ├── ostrack_256.onnx
│   └── osnet_ain_x0_5.onnx
│
├── OSTrack/
│
├── target_input_images/
│
├── videos/
│
├── yolo_env/
│
├── CMakeLists.txt
├── convert_models.py
└── README.md
```

---

# Dependencies

The project was developed and tested using the following software.

| Component | Version |
|-----------|---------|
| C++ | C++17 |
| Visual Studio | 2022 |
| CMake | 3.20+ |
| OpenCV | 4.x |
| ONNX Runtime | 1.22.0 |
| CUDA | 12.x |

---

# Required Models

The project requires the following pretrained models.

## Original Models (`models/`)

| Model | Purpose | Download |
|------|---------|----------|
| **YOLO11s** (`yolo11s.pt`) | Object Detection | https://docs.ultralytics.com/models/yolo11/ |
| **OSTrack** (`OSTrack_ep0300.pth.tar`) | Single Object Tracking | https://github.com/botaoye/OSTrack |
| **OSNet** (`osnet_ain_x0_5_imagenet.pth`) | Person Re-Identification | https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO |

---

## ONNX Models (`models_onnx/`)

The C++ application performs inference using ONNX models.

Place the exported models inside the **models_onnx/** directory.

```text
models_onnx/
├── yolo11s.onnx
├── ostrack_256.onnx
└── osnet_ain_x0_5.onnx
```
## Model Conversion 
The repository also contains **convert_models.py**, which can be used to export the original PyTorch models to ONNX format if necessary.
To keep this repository lightweight, the OSTrack repository and its configuration files (.yaml) are not bundled in this repository.
If you need to re-export ostrack_256.onnx using convert_models.py, please clone the official OSTrack repository to access the required config/yaml files:
Clone OSTrack repository for configuration files and model definitions
git clone [https://github.com/botaoye/OSTrack.git](https://github.com/botaoye/OSTrack.git)

> **Note:** The original PyTorch weights are provided for reproducibility and model conversion purposes. The application itself performs inference using the exported ONNX models.


---

# Build

```bash
mkdir build
cd build

cmake ..
cmake --build . --config Release
```

---

# Run

```bash
SingleInstanceDetection.exe
```

Before running the application:

- Place the original model weights inside **models/** (if model conversion is required).
- Place the exported ONNX models inside **models_onnx/**.
- Place the reference target images inside **target_input_images/**.
- Place the test videos inside **videos/**.

---


# Additional Experiments

Several alternative approaches were explored during the development of this case study.

### ORB Feature Matching

ORB descriptors were evaluated as a lightweight appearance matching method. Although computationally efficient, they were not sufficiently reliable for small UAV targets due to the limited number of stable keypoints.

### Motion-Based Search Prediction

A simple velocity-based prediction strategy was investigated to estimate the target location after temporary tracking loss. While it worked reasonably well in some occlusion scenarios, its performance was not consistent enough for the final implementation.

### OSNet for Vehicle Matching

The pretrained OSNet model was also evaluated for vehicle appearance matching. Since the available model was trained specifically for person re-identification, it did not produce sufficiently discriminative embeddings for vehicle instances.

These experiments helped guide the design decisions adopted in the final pipeline.

---

# Future Work

Possible future improvements include:

- Evaluating dedicated Vehicle Re-ID models.
- Benchmarking the pipeline on NVIDIA Jetson edge platforms.
- Investigating more robust target reacquisition strategies.
- Evaluating the system on a wider range of challenging UAV scenarios.

---


# Notes

This project was developed as part of a technical case study on Single Instance Object Detection.

The goal was to build a working pipeline that combines object detection, appearance matching and visual tracking in a modular C++ application suitable for real-time edge deployment.