# E57 Image Extraction Module

## Product Tagging in VEO360 Digital Twin Using Object Recognition

This module is the first stage of the VEO360 Product Tagging pipeline developed for the VEO Challenge. Its purpose is to extract visual information from E57 point cloud scans and convert it into images that can be used by downstream AI models for object detection, OCR, and digital twin asset tagging.

---

## Purpose

Industrial digital twins often require manual asset tagging, which can be:

- Time-consuming
- Error-prone
- Difficult to scale

This script automates the extraction of image data from Matterport and other E57-based scans, providing the visual input required for intelligent asset recognition and automated tagging workflows.

---

# Position in the Overall Pipeline

```text
Clean Reference Image (ABB REX615)
                │
                ▼
Synthetic Data Generator
                │
                ▼
YOLOv8 + OCR Training
                │
                ▼
        E57 Scan Processing
                │
                ▼
      01_extract_e57.py
                │
                ▼
     Panorama/Image Dataset
                │
                ▼
      Detection & OCR
                │
                ▼
    Interactive Asset Tags
                │
                ▼
      Document Retrieval
```

This module corresponds to the **Digital Twin Extraction** stage of the overall architecture.

---

# What This Script Does

The script performs the following tasks:

### 1. Opens an E57 Scan File

The script loads an E57 file using the `pye57` library.

Example:

```python
e57 = pye57.E57(input_file)
```

The file may contain:

- Point cloud data
- RGB color information
- Embedded scan images
- Scan metadata

---

### 2. Extracts Embedded Images

If images already exist inside the E57 scan, the script extracts them.

Supported image types:

```text
JPEG
PNG
```

Supported representations:

```text
sphericalRepresentation
pinholeRepresentation
cylindricalRepresentation
```

Output example:

```text
scan_image_0_sphericalRepresentation.jpg
scan_image_1_pinholeRepresentation.png
```

---

### 3. Generates Panoramas From Point Clouds

If no embedded images are available, the script automatically creates panoramic images from the point cloud data.

The process:

```text
Point Cloud
     ↓
Cartesian Coordinates
     ↓
Spherical Projection
     ↓
Equirectangular Panorama
     ↓
Image Enhancement
```

---

### 4. Creates AI-Ready Images

The generated panoramas are later consumed by:

- YOLOv8 Object Detection
- OCR Pipelines
- Asset Classification Models
- Digital Twin Tagging Systems

These outputs become the input for the object recognition stage of the VEO360 solution.

---

# Prerequisites

## Python Version

Recommended:

```text
Python 3.9+
```

## Dependencies

Install required packages:

```bash
pip install numpy pye57 opencv-python
```

Additional project dependencies used by the full solution:

```bash
pip install ultralytics albumentations easyocr pillow
```

These packages support object detection, synthetic image generation, and OCR workflows.

---

# Input

The script accepts an E57 scan file.

Example:

```text
cloud_0.e57
factory_scan.e57
substation.e57
```

---

# Running the Script

## Command

```bash
python scripts/01_extract_e57.py \
    --input data/scans/cloud_0.e57 \
    --output data/scans/extracted/
```

Example:

```bash
python scripts/01_extract_e57.py \
    --input scans/cloud_0.e57 \
    --output scans/extracted/
```

This command extracts embedded images or generates panoramas from the scan data.

---

# Internal Processing Workflow

## Step 1: Read Scan

```python
scan = e57.read_scan(scan_index)
```

Extracts:

```python
cartesianX
cartesianY
cartesianZ

colorRed
colorGreen
colorBlue
```

---

## Step 2: Convert Coordinates

Cartesian coordinates:

```python
X, Y, Z
```

are converted into spherical coordinates:

```python
theta
phi
radius
```

for panoramic projection.

---

## Step 3: Project Points

Points are mapped to image coordinates.

Default panorama size:

```text
2048 × 1024
```

Result:

```text
Equirectangular Panorama
```

---

## Step 4: Fill Empty Pixels

OpenCV inpainting is applied:

```python
cv2.inpaint()
```

to reduce projection gaps and create smoother panoramas.

---

## Step 5: Save Outputs

Generated files are written to the specified output directory.

Example:

```text
data/scans/extracted/
│
├── scan_image_0_sphericalRepresentation.jpg
├── scan_image_1_pinholeRepresentation.jpg
├── projected_scan_0.jpg
└── projected_scan_1.jpg
```

---

# Output

The module produces:

### Extracted Images

```text
*.jpg
*.png
```

### Generated Panoramas

```text
projected_scan_*.jpg
```

These outputs are later used by:

- 02_generate_synth.py
- 03_train_detector.py
- 04_infer_and_tag.py

within the complete VEO360 workflow.

---

# Relationship With Other Modules

## 01_extract_e57.py

Extracts images from E57 scans.

### Output

```text
Panorama Images
```

↓

## 02_generate_synth.py

Creates synthetic training data through:

- Lighting variation
- Occlusions
- Perspective transformations
- Background blending

↓

## 03_train_detector.py

Fine-tunes a YOLOv8 model on generated datasets.

↓

## 04_infer_and_tag.py

Performs:

- Asset Detection
- OCR
- Equipment Identification
- Coordinate Mapping
- Documentation Linking

↓

## VEO360 Digital Twin

Automatically generates interactive asset tags linked to equipment manuals and maintenance records.

---

# Example End-to-End Workflow

```text
cloud_0.e57
        │
        ▼
01_extract_e57.py
        │
        ▼
Panorama Images
        │
        ▼
YOLOv8 Detection
        │
        ▼
OCR Extraction
        │
        ▼
ABB REX615 Recognition
        │
        ▼
Asset Documentation Match
        │
        ▼
Interactive 3D VEO360 Tag
```

---

# Expected Benefits

- Eliminates manual extraction of scan imagery.
- Produces AI-ready image datasets.
- Supports synthetic data generation.
- Enables automated asset recognition.
- Accelerates VEO360 digital twin deployment.
- Improves access to maintenance documentation.

---

# Module Information

Developed as part of the VEO Hackathon Challenge.

Module:

```text
scripts/01_extract_e57.py
```

Role:

```text
Digital Twin Image Extraction Layer
```

within the VEO360 Product Tagging Solution.
