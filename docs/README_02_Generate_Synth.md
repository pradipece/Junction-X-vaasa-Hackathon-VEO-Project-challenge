# Synthetic Dataset Generation Module

## Product Tagging in VEO360 Digital Twin Using Object Recognition

This module generates synthetic training data from reference product images. It is the second stage of the VEO360 Product Tagging pipeline and is responsible for creating large-scale annotated datasets that can be used to train object detection models when real-world labeled data is limited.

The generated dataset follows the YOLO format and includes both training images and bounding-box annotations.

---

# Purpose

Obtaining thousands of labeled industrial equipment images is often difficult and expensive.

This module solves that problem by:

- Taking one or more reference product images
- Applying realistic visual variations
- Placing products onto industrial-like backgrounds
- Generating automatic YOLO annotations
- Producing a complete training dataset

The resulting dataset can be used to train asset recognition models for Digital Twin environments.

---

# Position in the Overall Pipeline

```text
E57 Scan Data
      │
      ▼
01_extract_e57.py
      │
      ▼
Panorama Images
      │
      ▼
02_generate_synth.py
      │
      ▼
Synthetic Training Dataset
      │
      ▼
03_train_detector.py
      │
      ▼
YOLOv8 Detection Model
      │
      ▼
04_infer_and_tag.py
      │
      ▼
Asset Detection & OCR
      │
      ▼
VEO360 Interactive Tags
```

This module transforms a small number of product reference images into hundreds or thousands of training samples.

---

# What This Script Does

The script automatically creates synthetic images using several augmentation and domain-randomization techniques.

## Features

### Industrial Background Generation

If background images are not provided, the script generates artificial industrial-looking backgrounds.

The synthetic background includes:

- Random base colors
- Surface noise
- Panel-like structures
- Divider lines

This simulates electrical cabinets and industrial control panels commonly found in Digital Twin environments.

---

### Automatic Object Segmentation

The script extracts the foreground object from the reference image.

Supported formats:

```text
PNG
JPG
JPEG
```

For PNG images:

```text
Alpha channel is used as the mask.
```

For JPG images:

```text
White backgrounds are automatically removed.
```

The result is an object mask that enables realistic placement on new backgrounds.

---

### Domain Randomization

To improve model robustness, each generated image receives randomized transformations.

Applied augmentations include:

#### Perspective Transformation

Simulates different viewing angles.

```text
Front View
Tilted View
Side View
Matterport-like Perspective
```

#### Brightness Variation

Randomly increases or decreases brightness.

```text
Dark Environment
Normal Lighting
Bright Environment
```

#### Contrast Variation

Adjusts image contrast to represent different camera conditions.

#### Blur Effects

Simulates:

- Motion blur
- Focus variation
- Scan imperfections

#### Occlusions

Random rectangular obstacles are added.

Examples:

```text
Cables
Labels
Wires
Small Objects
Wear Effects
```

This helps the detector learn to identify partially visible assets.

---

### Object Scaling

Each product is resized before placement.

Typical range:

```python
0.25 – 0.65
```

of the image width.

This simulates assets appearing at different distances from the camera.

---

### Random Placement

The object is inserted at a random location within the generated scene.

Benefits:

- Position invariance
- Better detector generalization
- Increased training diversity

---

### Automatic YOLO Label Generation

The script automatically computes object locations and creates YOLO annotation files.

Format:

```text
class_id
x_center
y_center
width
height
```

Example:

```text
0 0.523412 0.441500 0.210223 0.317800
```

Current class mapping:

```text
0 = REX615
```

---

# Prerequisites

## Python Version

Recommended:

```text
Python 3.9+
```

---

# Required Libraries

Install dependencies:

```bash
pip install numpy opencv-python
```

Additional project dependencies:

```bash
pip install ultralytics albumentations pillow
```

---

# Input Requirements

## Reference Images

The reference directory contains object images.

Example:

```text
data/reference/
│
├── rex615.png
├── rex615_front.png
└── rex615_side.jpg
```

These images represent the product that will be inserted into synthetic environments.

---

## Optional Background Images

The script can also accept background photographs.

Example:

```text
data/backgrounds/
│
├── cabinet.jpg
├── panel.jpg
└── substation.jpg
```

If no backgrounds are provided, synthetic industrial backgrounds will be created automatically.

---

# Running the Script

## Basic Usage

```bash
python 02_generate_synth.py \
    --ref-dir data/reference \
    --output data/synthetic \
    --count 500
```

---

## Using Custom Backgrounds

```bash
python 02_generate_synth.py \
    --ref-dir data/reference \
    --bg-dir data/backgrounds \
    --output data/synthetic \
    --count 1000
```

---

# Command Line Parameters

## --ref-dir

Required.

Path to reference images.

Example:

```bash
--ref-dir data/reference
```

---

## --bg-dir

Optional.

Path to industrial background images.

Example:

```bash
--bg-dir data/backgrounds
```

---

## --output

Output directory.

Default:

```bash
data/synthetic
```

---

## --count

Number of synthetic samples to generate.

Default:

```bash
500
```

Example:

```bash
--count 1000
```

---

# Internal Processing Workflow

## Step 1

Load Reference Image

```python
load_foreground_with_mask()
```

Output:

```text
Foreground Image
+
Alpha Mask
```

---

## Step 2

Load Background

Options:

```text
Real Industrial Photo
OR
Generated Industrial Background
```

---

## Step 3

Apply Domain Randomization

```python
apply_domain_randomization()
```

Operations:

- Perspective Warp
- Blur
- Brightness Shift
- Contrast Shift
- Occlusion Generation

---

## Step 4

Scale Product

Random scaling:

```python
0.25 → 0.65
```

This creates varying object sizes.

---

## Step 5

Composite Object

The object is blended into the scene using alpha masking.

```python
mask_norm
```

controls object transparency and edge quality.

---

## Step 6

Calculate Bounding Box

The script computes:

```text
x_center
y_center
width
height
```

using normalized YOLO coordinates.

---

## Step 7

Save Training Data

Generated output:

```text
Image
+
Label File
```

Example:

```text
synth_rex615_00001.jpg
synth_rex615_00001.txt
```

---

## Step 8

Create YOLO Dataset Configuration

The script automatically generates:

```text
data.yaml
```

Contents:

```yaml
path: data/synthetic

train: images/train
val: images/train

names:
  0: REX615
```

This file can immediately be used by YOLOv8 training.

---

# Output Structure

```text
data/synthetic/
│
├── data.yaml
│
├── images/
│   └── train/
│
│       ├── synth_rex615_00000.jpg
│       ├── synth_rex615_00001.jpg
│       └── ...
│
└── labels/
    └── train/

        ├── synth_rex615_00000.txt
        ├── synth_rex615_00001.txt
        └── ...
```

---

# Example End-to-End Workflow

```text
Reference Product Images
            │
            ▼
02_generate_synth.py
            │
            ▼
1000 Synthetic Images
            │
            ▼
YOLO Labels
            │
            ▼
YOLO Dataset
            │
            ▼
03_train_detector.py
            │
            ▼
Custom Asset Detector
            │
            ▼
Asset Recognition in Digital Twins
```

---

# Benefits

## Reduced Data Collection Costs

A small number of reference images can produce thousands of training samples.

## Improved Model Robustness

Domain randomization improves performance in real-world environments.

## Automatic Annotation

No manual bounding-box labeling is required.

## Faster Development

The dataset is immediately compatible with YOLOv8.

## Better Generalization

The model learns to recognize equipment under varying:

- Lighting conditions
- Viewing angles
- Occlusions
- Perspective distortions

---

# Generated Dataset

The script currently generates labels for:

```text
Class 0 = REX615
```

Additional products can be added by extending the class definitions and dataset configuration.

---

# Module Information

Developed as part of the VEO360 Product Tagging Challenge.

Module:

```text
02_generate_synth.py
```

Role:

```text
Synthetic Training Data Generation Layer
```

within the VEO360 Digital Twin Asset Recognition Pipeline.
