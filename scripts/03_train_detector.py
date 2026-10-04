import argparse
import os
import sys
import torch
from ultralytics import YOLO


def train_detector(
    data_yaml: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 16,
    model_name: str = "yolov8s.pt",
    project_dir: str = "runs/train",
    experiment_name: str = "rex615_detector",
):
    """Fine-tunes a pre-trained YOLOv8 model using synthetic industrial dataset."""
    if not os.path.exists(data_yaml):
        print(f"[-] Error: Dataset config file not found at '{data_yaml}'.")
        print(
            "    Please run scripts/02_generate_synth.py first to create the synthetic data."
        )
        sys.exit(1)

    # Hardware detection
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {'CUDA GPU' if device == '0' else 'CPU'}")
    if device == "cpu":
        print("[!] Warning: Training on CPU. Reduced batch size recommended.")
        batch_size = min(batch_size, 8)

    print(f"[*] Loading pre-trained base model: {model_name}")
    model = YOLO(model_name)

    print(
        f"[*] Starting fine-tuning for {epochs} epochs (Image Size: {imgsz}, Batch: {batch_size})..."
    )

    try:
        # Fine-tune with data augmentations suitable for industrial domains
        results = model.train(
            data=os.path.abspath(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project_dir,
            name=experiment_name,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.2,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            verbose=True,
        )

        best_weight_path = os.path.join(
            project_dir, experiment_name, "weights", "best.pt"
        )
        print("\n" + "=" * 60)
        print("[✓] Model Training Completed!")
        print(f"[✓] Best weights saved to: {best_weight_path}")
        print("=" * 60)

        # Run validation metrics summary
        metrics = model.val()
        print(
            f"[*] Validation mAP@50: {metrics.box.map50:.4f} | mAP@50-95: {metrics.box.map:.4f}"
        )

    except Exception as e:
        print(f"[-] Training encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on synthetic industrial data for VEO360."
    )
    parser.add_argument(
        "--data",
        default="data/synthetic/data.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30 for quick hackathon turnaround)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8s.pt",
        help="Base model checkpoint (yolov8n.pt or yolov8s.pt)",
    )
    parser.add_argument(
        "--project",
        default="models",
        help="Target folder to save weights and logs",
    )
    parser.add_argument(
        "--name",
        default="rex615_detector",
        help="Experiment name directory",
    )

    args = parser.parse_args()
    train_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        model_name=args.base_model,
        project_dir=args.project,
        experiment_name=args.name,
    )