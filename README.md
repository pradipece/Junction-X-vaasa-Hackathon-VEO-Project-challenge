# Product Tagging in VEO360 Digital Twin Using Object Recognition - Project Hail Mary

An automated computer vision and document linking pipeline developed for the VEO Challenge. This project bridges the domain gap between clean master product imagery and complex industrial point cloud environments to enable automated, interactive asset tagging in digital twins.

---

## 📌 Project Overview

- **Business Problem:** Industrial digital twins rely heavily on manual tagging, leading to human error, slow deployments, and high operational costs. A lack of real-world labeled industrial datasets further restricts scalable computer vision deployment.
- **Core Solution:** An end-to-end pipeline that ingests clean reference product imagery, synthesizes domain-randomized training data, trains robust object detection/OCR models, and automatically locates assets inside 3D scan environments to link technical documentation.
- **Impact:** Decreases site preparation overhead, automates maintenance documentation access, and accelerates digital twin scalability across industrial sites.

---

## 🏗️ Architecture & Pipeline Workflow

```text
[Clean Reference Image (REX615)]
            │
            ▼
[Synthetic Data Generator (Domain Randomization)]
            │
            ▼
[Fine-Tuned Detector (YOLOv8 + OCR)]
            │
            ▼
[Matterport Scan Panorama (cloud_0.e57)] ──► [Interactive 3D Tag & Document Retrieval]
```

1. **Synthetic Data Generation:** Augments master reference images of the ABB REX615 relay through affine transformations, simulated lens artifacts, lighting variations, occlusions, and background blending to match field conditions.
2. **Model Training & Text Extraction:** Trains a YOLOv8 detector on synthetic data to locate the relay body and panel labels, coupled with OCR to read asset identifiers (e.g., "615", "ABB").
3. **Digital Twin Extraction:** Parses `cloud_0.e57` point cloud files to extract 2D equirectangular/panoramic scan perspectives.
4. **Asset Matching & 3D Tagging:** Matches extracted asset text to technical datasheets/manuals and computes 3D anchor coordinates inside the digital twin environment.

---

## 📁 Repository Structure

```text
├── data/
│   ├── reference/             # Master product renders (ABB REX615)
│   ├── background/            # Industrial backdrop images for synthetic compositing
│   ├── synthetic/             # Generated dataset with YOLO format annotations
│   └── scans/                 # Extracted panoramas from cloud_0.e57
├── scripts/
│   ├── 01_extract_e57.py      # Extracts 2D imagery/panoramas from .e57 files
│   ├── 02_generate_synth.py   # Domain randomization & synthetic dataset builder
│   ├── 03_train_detector.py   # Model fine-tuning script
│   └── 04_infer_and_tag.py    # Detection, OCR, and coordinate mapping
├── docs/                      # Technical manuals repository & assets
├── web_viewer/                # Prototype interactive 3D digital twin viewer
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/veo360-product-tagging.git
cd veo360-product-tagging

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install ultralytics albumentations opencv-python numpy pye57 easyocr pillow
```

---

## 🚀 Usage

**Step 1: Extract images from `.e57` scans**

```bash
python scripts/01_extract_e57.py --input data/scans/cloud_0.e57 --output data/scans/extracted/
```

**Step 2: Generate synthetic dataset**

```bash
python scripts/02_generate_synth.py --ref-dir data/reference/ --bg-dir data/background/ --output data/synthetic/ --count 1000
```

**Step 3: Train detection model**

```bash
python scripts/03_train_detector.py --data data/synthetic/data.yaml --epochs 50 --imgsz 640
```

**Step 4: Run inference and output tags**

```bash
python scripts/04_infer_and_tag.py --weights models/best.pt --input data/scans/extracted/ --docs-registry docs/asset_registry.json --output web_viewer/tags.json
```

---

## 🖥️ Demo & Results

- **Synthetic Data Generation:** Transformed a single clean master image of the ABB REX615 into 1,000+ realistic industrial training samples covering occlusions, reflections, and varied angles.
- **Real Data Generalization:** Demonstrated robust detection accuracy on real Matterport scan frames without requiring manual annotations on the actual site.
- **Interactive Digital Twin Tagging:** Clicking on an automatically detected ABB REX615 unit presents an interactive 3D pin displaying equipment specifications, operating status, and a direct link to the maintenance manual.

---

## 👥 Contributors

Developed as part of the VEO Hackathon Challenge.

---

## 👥 Authors

- Name: **Pradip Nath**, **Thang Ngo**
