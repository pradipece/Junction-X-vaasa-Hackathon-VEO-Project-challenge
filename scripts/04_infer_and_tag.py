import argparse
import glob
import json
import math
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


DEFAULT_REGISTRY = {
    "REX615": {
        "asset_name": "ABB Relion REX615 Protection & Control Relay",
        "category": "Medium-Voltage Protection",
        "manufacturer": "ABB",
        "documentation_url": "https://search.abb.com/library/Download.aspx?DocumentID=1MRS756378",
        "wiring_diagram_url": "https://library.e.abb.com/public/rex615_wiring.pdf",
        "status": "Operational",
        "last_inspection": "2026-03-15",
    },
    "DEFAULT": {
        "asset_name": "Generic Switchgear Asset",
        "category": "Industrial Electrical",
        "manufacturer": "Unknown",
        "documentation_url": "https://veo.fi/solutions/",
        "status": "Under Review",
    },
}


def load_doc_registry(registry_path: str) -> dict:
    if registry_path and os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_REGISTRY


def pixel_to_spherical_coordinates(
    x_center: float, y_center: float, img_w: int, img_h: int, assumed_depth: float = 2.5
) -> dict:
    """
    Projects 2D equirectangular pixel coordinates (u, v) into 3D camera-relative coordinates (X, Y, Z).
    """
    # Longitude theta [-pi, pi], Latitude phi [-pi/2, pi/2]
    theta = (x_center / img_w) * 2.0 * math.pi - math.pi
    phi = (0.5 - (y_center / img_h)) * math.pi

    # Spherical to Cartesian
    x = assumed_depth * math.cos(phi) * math.sin(theta)
    y = assumed_depth * math.sin(phi)
    z = assumed_depth * math.cos(phi) * math.cos(theta)

    return {
        "spherical": {
            "yaw_deg": round(math.degrees(theta), 2),
            "pitch_deg": round(math.degrees(phi), 2),
        },
        "cartesian_3d": {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "estimated_distance_m": assumed_depth,
        },
    }


def run_pipeline(
    weights_path: str,
    input_dir: str,
    output_json: str,
    output_vis_dir: str,
    registry_path: str,
    conf_thresh: float = 0.35,
):
    if not os.path.exists(weights_path):
        print(f"[-] Error: Weights file not found at '{weights_path}'")
        sys.exit(1)

    scan_files = glob.glob(os.path.join(input_dir, "*.*"))
    scan_files = [
        f for f in scan_files if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not scan_files:
        print(f"[-] No scan images found in '{input_dir}'")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    registry = load_doc_registry(registry_path)
    print(f"[*] Loaded asset documentation registry with {len(registry)} keys.")

    print(f"[*] Loading detection model from '{weights_path}'...")
    model = YOLO(weights_path)

    ocr_reader = None
    if HAS_EASYOCR:
        print("[*] Initializing EasyOCR reader...")
        ocr_reader = easyocr.Reader(["en"], gpu=True)
    else:
        print("[!] EasyOCR not installed. Fallback to direct model label classification.")

    digital_twin_tags = []

    for scan_path in scan_files:
        img_name = os.path.basename(scan_path)
        print(f"[*] Processing scan: {img_name}")
        img = cv2.imread(scan_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        results = model.predict(source=img, conf=conf_thresh, verbose=False)[0]

        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detected_class = model.names[cls_id]

            # Crop region of interest for OCR inspection
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            roi = img[y1_c:y2_c, x1_c:x2_c]

            matched_key = detected_class
            ocr_text = []

            if ocr_reader and roi.size > 0:
                ocr_res = ocr_reader.readtext(roi, detail=0)
                ocr_text = [t.strip().upper() for t in ocr_res if len(t.strip()) > 1]
                for text_token in ocr_text:
                    if "615" in text_token or "REX" in text_token:
                        matched_key = "REX615"
                        break

            # Retrieve documentation and metadata
            doc_data = registry.get(matched_key, registry.get("DEFAULT"))

            # Calculate 3D spatial anchor for VEO360 digital twin placement
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            spatial_anchor = pixel_to_spherical_coordinates(cx, cy, w, h)

            tag_id = f"TAG-{len(digital_twin_tags) + 1:04d}"
            tag_payload = {
                "tag_id": tag_id,
                "source_scan": img_name,
                "asset_key": matched_key,
                "confidence": round(conf, 3),
                "2d_bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "spatial_anchor_3d": spatial_anchor,
                "extracted_ocr_tokens": ocr_text,
                "metadata": doc_data,
            }
            digital_twin_tags.append(tag_payload)

            # Draw visual detection and tag indicator
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = f"[{tag_id}] {matched_key} ({conf:.2f})"
            cv2.putText(
                img,
                label_text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Save annotated visualization
        vis_out = os.path.join(output_vis_dir, f"tagged_{img_name}")
        cv2.imwrite(vis_out, img)
        print(f"[+] Saved tagged visual overlay: {vis_out}")

    # Export structured JSON for the digital twin viewer
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "project": "VEO360 Digital Twin Intelligent Tagging",
                "total_tags_created": len(digital_twin_tags),
                "tags": digital_twin_tags,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(f"[✓] Pipeline complete! Generated {len(digital_twin_tags)} intelligent tags.")
    print(f"[✓] Digital Twin Tag Data: {output_json}")
    print(f"[✓] Annotated Overlays: {output_vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run detection, OCR, and generate 3D digital twin tags linked to documentation."
    )
    parser.add_argument(
        "--weights",
        default="models/rex615_detector/weights/best.pt",
        help="Path to trained YOLOv8 model weights",
    )
    parser.add_argument(
        "--input",
        default="data/scans/extracted",
        help="Directory containing extracted scan panorama images",
    )
    parser.add_argument(
        "--docs-registry",
        default="docs/asset_registry.json",
        help="JSON file containing product metadata and manual links",
    )
    parser.add_argument(
        "--output",
        default="web_viewer/tags.json",
        help="Output JSON file for digital twin integration",
    )
    parser.add_argument(
        "--vis-dir",
        default="data/scans/annotated",
        help="Directory to save annotated visualization images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detection",
    )

    args = parser.parse_args()
    run_pipeline(
        weights_path=args.weights,
        input_dir=args.input,
        output_json=args.output,
        output_vis_dir=args.vis_dir,
        registry_path=args.docs_registry,
        conf_thresh=args.conf,
    )