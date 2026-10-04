import argparse
import glob
import math
import os
import random
import cv2
import numpy as np


def ensure_dirs(output_dir: str):
    images_dir = os.path.join(output_dir, "images", "train")
    labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def create_fallback_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Creates a synthetic gradient/noisy industrial panel background if no backdrops are provided."""
    base_color = np.random.randint(40, 180, size=(1, 1, 3), dtype=np.uint8)
    bg = np.full((height, width, 3), base_color, dtype=np.uint8)
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add random panel divider lines
    for _ in range(random.randint(2, 6)):
        pt1 = (random.randint(0, width), 0)
        pt2 = (random.randint(0, width), height)
        cv2.line(bg, pt1, pt2, (30, 30, 30), thickness=random.randint(2, 5))
    return bg


def load_foreground_with_mask(image_path: str):
    """Loads reference image and creates an alpha mask (cutting out white/light borders if needed)."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        # Automatic mask for white background studio images
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    return bgr, alpha


def apply_domain_randomization(fg_bgr: np.ndarray, fg_mask: np.ndarray):
    """Applies lighting, perspective tilt, blur, noise, and occlusions."""
    h, w = fg_bgr.shape[:2]

    # 1. Perspective Warp (simulates viewing angle from Matterport scan)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = 0.15
    dst_pts = np.float32(
        [
            [random.uniform(0, w * jitter), random.uniform(0, h * jitter)],
            [w - random.uniform(0, w * jitter), random.uniform(0, h * jitter)],
            [w - random.uniform(0, w * jitter), h - random.uniform(0, h * jitter)],
            [random.uniform(0, w * jitter), h - random.uniform(0, h * jitter)],
        ]
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    fg_bgr = cv2.warpPerspective(fg_bgr, M, (w, h), borderValue=(0, 0, 0))
    fg_mask = cv2.warpPerspective(fg_mask, M, (w, h), borderValue=0)

    # 2. Lighting & Color Jitter
    alpha_contrast = random.uniform(0.7, 1.3)
    beta_brightness = random.randint(-40, 40)
    fg_bgr = np.clip(alpha_contrast * fg_bgr + beta_brightness, 0, 255).astype(np.uint8)

    # 3. Occasional blur (depth of field / motion blur)
    if random.random() < 0.4:
        k_size = random.choice([3, 5])
        fg_bgr = cv2.GaussianBlur(fg_bgr, (k_size, k_size), 0)

    # 4. Random scratches/occlusion rectangles (simulating wires, tags, wear)
    if random.random() < 0.5:
        for _ in range(random.randint(1, 3)):
            occ_x1 = random.randint(0, w - 30)
            occ_y1 = random.randint(0, h - 30)
            occ_w = random.randint(15, 80)
            occ_h = random.randint(15, 80)
            color = (
                random.randint(20, 60),
                random.randint(20, 60),
                random.randint(20, 60),
            )
            cv2.rectangle(
                fg_bgr,
                (occ_x1, occ_y1),
                (occ_x1 + occ_w, occ_y1 + occ_h),
                color,
                -1,
            )

    return fg_bgr, fg_mask


def generate_synthetic_samples(
    ref_dir: str, bg_dir: str, output_dir: str, num_samples: int = 1000
):
    images_dir, labels_dir = ensure_dirs(output_dir)

    ref_images = glob.glob(os.path.join(ref_dir, "*.*"))
    ref_images = [
        f for f in ref_images if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not ref_images:
        print(f"[-] No reference images found in {ref_dir}")
        return

    bg_images = glob.glob(os.path.join(bg_dir, "*.*")) if bg_dir else []
    bg_images = [
        f for f in bg_images if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(
        f"[*] Starting synthetic generation: {num_samples} samples from {len(ref_images)} reference image(s)."
    )

    for i in range(num_samples):
        # Pick reference asset
        ref_path = random.choice(ref_images)
        fg_bgr, fg_mask = load_foreground_with_mask(ref_path)
        if fg_bgr is None:
            continue

        # Load or create background canvas
        if bg_images and random.random() < 0.7:
            bg_path = random.choice(bg_images)
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, (1280, 720))
        else:
            bg = create_fallback_background(1280, 720)

        canvas_h, canvas_w = bg.shape[:2]

        # Apply augmentation & domain randomization to the device
        fg_bgr, fg_mask = apply_domain_randomization(fg_bgr, fg_mask)

        # Scale foreground to a realistic perspective ratio inside the scene
        scale = random.uniform(0.25, 0.65)
        new_w = max(40, int(canvas_w * scale))
        new_h = int(fg_bgr.shape[0] * (new_w / fg_bgr.shape[1]))

        if new_h >= canvas_h or new_w >= canvas_w:
            continue

        fg_resized = cv2.resize(fg_bgr, (new_w, new_h))
        mask_resized = cv2.resize(fg_mask, (new_w, new_h))

        # Random placement on background
        pos_x = random.randint(0, canvas_w - new_w)
        pos_y = random.randint(0, canvas_h - new_h)

        # Alpha composite overlay
        roi = bg[pos_y : pos_y + new_h, pos_x : pos_x + new_w]
        mask_norm = (mask_resized.astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended = (
            fg_resized.astype(np.float32) * mask_norm
            + roi.astype(np.float32) * (1.0 - mask_norm)
        ).astype(np.uint8)

        bg[pos_y : pos_y + new_h, pos_x : pos_x + new_w] = blended

        # Compute YOLO normalized coordinates [class_id, x_center, y_center, width, height]
        # Class 0: REX615
        bbox_x_center = (pos_x + new_w / 2.0) / canvas_w
        bbox_y_center = (pos_y + new_h / 2.0) / canvas_h
        bbox_w = new_w / canvas_w
        bbox_h = new_h / canvas_h

        sample_name = f"synth_rex615_{i:05d}"
        img_out_path = os.path.join(images_dir, f"{sample_name}.jpg")
        txt_out_path = os.path.join(labels_dir, f"{sample_name}.txt")

        cv2.imwrite(img_out_path, bg)
        with open(txt_out_path, "w") as f_label:
            f_label.write(
                f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n"
            )

        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            print(f"[+] Generated {i + 1}/{num_samples} images...")

    # Write YOLO dataset configuration YAML
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f_yaml:
        f_yaml.write(
            f"path: {os.path.abspath(output_dir)}\n"
            f"train: images/train\n"
            f"val: images/train\n\n"
            f"names:\n"
            f"  0: REX615\n"
        )
    print(f"\n[✓] Synthetic data generation complete!")
    print(f"[✓] YOLO data definition saved to: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from master reference images."
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        help="Path to reference master images (e.g. data/reference/)",
    )
    parser.add_argument(
        "--bg-dir",
        default="",
        help="Optional path to directory of industrial background photos",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic",
        help="Target output directory for images and YOLO labels",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of synthetic samples to synthesize",
    )

    args = parser.parse_args()
    generate_synthetic_samples(
        ref_dir=args.ref_dir,
        bg_dir=args.bg_dir,
        output_dir=args.output,
        num_samples=args.count,
    )