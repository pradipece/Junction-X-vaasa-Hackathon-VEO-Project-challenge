import argparse
import os
import subprocess
import sys
import time


def print_step_banner(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f" [STEP {step_num}/4] {title.upper()}")
    print("=" * 70)


def execute_command(cmd_args: list, step_name: str):
    """Executes a subprocess command and halts if any step fails."""
    start_time = time.time()
    cmd_str = " ".join(cmd_args)
    print(f"[*] Executing: {cmd_str}\n")

    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Stream real-time script output to terminal
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    process.stdout.close()
    return_code = process.wait()

    elapsed = time.time() - start_time
    if return_code != 0:
        print(f"\n[-] Execution failed at {step_name} with exit code {return_code}.")
        sys.exit(return_code)

    print(f"\n[✓] {step_name} completed successfully in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for VEO360 Product Tagging Pipeline."
    )
    # File paths
    parser.add_argument(
        "--e57-file",
        default="data/scans/cloud_0.e57",
        help="Path to input .e57 point cloud file",
    )
    parser.add_argument(
        "--ref-dir",
        default="data/reference",
        help="Directory containing clean master reference images (REX615)",
    )
    parser.add_argument(
        "--bg-dir",
        default="data/background",
        help="Optional directory containing background photos for synthetic generation",
    )
    parser.add_argument(
        "--docs-registry",
        default="docs/asset_registry.json",
        help="JSON mapping of asset IDs to documentation manuals",
    )

    # Output directories
    parser.add_argument(
        "--extracted-scans",
        default="data/scans/extracted",
        help="Directory to save extracted 2D scan images",
    )
    parser.add_argument(
        "--synthetic-dir",
        default="data/synthetic",
        help="Directory to save generated synthetic dataset",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store trained model checkpoints",
    )
    parser.add_argument(
        "--output-json",
        default="web_viewer/tags.json",
        help="Destination JSON for digital twin interactive tags",
    )
    parser.add_argument(
        "--annotated-dir",
        default="data/scans/annotated",
        help="Directory to save annotated scan overlay images",
    )

    # Hyperparameters
    parser.add_argument(
        "--synth-count",
        type=int,
        default=500,
        help="Number of synthetic samples to synthesize (default: 500)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for YOLOv8 (default: 30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model image resolution (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for inference (default: 0.35)",
    )

    # Flow control switches
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip E57 image extraction if already performed",
    )
    parser.add_argument(
        "--skip-synth",
        action="store_true",
        help="Skip synthetic dataset generation if already generated",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use existing model checkpoint",
    )

    args = parser.parse_args()
    python_bin = sys.executable
    total_start = time.time()

    print("=" * 70)
    print(" VEO360 DIGITAL TWIN AUTO-TAGGING: MASTER WORKFLOW")
    print("=" * 70)

    # STEP 1: Extract Scan Panoramas from E57
    print_step_banner(1, "Extract 2D Panoramas from Point Cloud")
    if args.skip_extraction:
        print("[!] --skip-extraction provided. Skipping step 1.")
    else:
        cmd_extract = [
            python_bin,
            "scripts/01_extract_e57.py",
            "--input",
            args.e57_file,
            "--output",
            args.extracted_scans,
        ]
        execute_command(cmd_extract, "Step 1: E57 Image Extraction")

    # STEP 2: Generate Synthetic Dataset with Domain Randomization
    print_step_banner(2, "Generate Synthetic Training Dataset")
    if args.skip_synth:
        print("[!] --skip-synth provided. Skipping step 2.")
    else:
        cmd_synth = [
            python_bin,
            "scripts/02_generate_synth.py",
            "--ref-dir",
            args.ref_dir,
            "--bg-dir",
            args.bg_dir,
            "--output",
            args.synthetic_dir,
            "--count",
            str(args.synth_count),
        ]
        execute_command(cmd_synth, "Step 2: Synthetic Data Generation")

    # STEP 3: Train Object Detection Model
    print_step_banner(3, "Fine-tune YOLOv8 on Synthetic Data")
    dataset_yaml = os.path.join(args.synthetic_dir, "data.yaml")
    trained_weights = os.path.join(args.models_dir, "rex615_detector", "weights", "best.pt")

    if args.skip_train:
        print("[!] --skip-train provided. Skipping step 3.")
        if not os.path.exists(trained_weights):
            print(f"[-] Warning: Expected checkpoint not found at '{trained_weights}'.")
    else:
        cmd_train = [
            python_bin,
            "scripts/03_train_detector.py",
            "--data",
            dataset_yaml,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--project",
            args.models_dir,
            "--name",
            "rex615_detector",
        ]
        execute_command(cmd_train, "Step 3: Model Training")

    # STEP 4: Run Inference, OCR, and Digital Twin Tagging
    print_step_banner(4, "Inference, OCR & Digital Twin Tag Generation")
    cmd_infer = [
        python_bin,
        "scripts/04_infer_and_tag.py",
        "--weights",
        trained_weights,
        "--input",
        args.extracted_scans,
        "--docs-registry",
        args.docs_registry,
        "--output",
        args.output_json,
        "--vis-dir",
        args.annotated_dir,
        "--conf",
        str(args.conf),
    ]
    execute_command(cmd_infer, "Step 4: Inference and Tag Generation")

    # Final summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f" Total execution time: {total_elapsed / 60:.2f} minutes")
    print(f" -> Output Digital Twin Tags : {os.path.abspath(args.output_json)}")
    print(f" -> Annotated Scan Panoramas : {os.path.abspath(args.annotated_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()