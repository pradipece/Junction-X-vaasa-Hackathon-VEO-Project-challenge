import argparse
import os
import sys
import numpy as np
import pye57


def extract_images_from_e57(e57_filepath: str, output_dir: str):
    """
    Parses an E57 file and extracts all embedded 2D image representations
    (spherical panoramas or pinhole/cylindrical scan images).
    """
    if not os.path.exists(e57_filepath):
        print(f"[-] Error: File not found at {e57_filepath}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Opening E57 point cloud: {e57_filepath}")

    try:
        e57 = pye57.E57(e57_filepath)
    except Exception as e:
        print(f"[-] Failed to open E57 file with pye57: {e}")
        sys.exit(1)

    root = e57.root
    image_count = 0

    # Standard E57 structure stores embedded 2D scans under 'images2D'
    if "images2D" in root:
        images_node = root["images2D"]
        total_images = len(images_node)
        print(f"[*] Found {total_images} 2D image entries inside 'images2D'.")

        for idx, img_entry in enumerate(images_node):
            name = f"scan_image_{idx}"
            if "name" in img_entry:
                name = img_entry["name"].value()

            # E57 representations: sphericalRepresentation, pinholeRepresentation, etc.
            blob_found = False
            for rep_type in [
                "sphericalRepresentation",
                "pinholeRepresentation",
                "cylindricalRepresentation",
            ]:
                if rep_type in img_entry:
                    rep = img_entry[rep_type]
                    # Check for JPEG or PNG image blobs
                    for img_format in [
                        "jpegImage",
                        "pngImage",
                        "imageMask",
                    ]:
                        if img_format in rep:
                            blob_node = rep[img_format]
                            data_bytes = bytearray(blob_node.read())
                            ext = "png" if "png" in img_format else "jpg"
                            filename = f"{name}_{rep_type}.{ext}"
                            out_path = os.path.join(output_dir, filename)

                            with open(out_path, "wb") as f:
                                f.write(data_bytes)

                            print(
                                f"[+] Extracted: {filename} ({len(data_bytes) / 1024:.1f} KB)"
                            )
                            image_count += 1
                            blob_found = True
                            break
                    if blob_found:
                        break

            if not blob_found:
                print(
                    f"[!] Warning: No raw image byte stream found for index {idx} ({name})."
                )

    # Fallback: if images2D is empty, project color point cloud data to panorama
    if image_count == 0:
        print("[!] No standard 'images2D' found. Attempting point cloud scan projection...")
        data3d = root["data3D"]
        print(f"[*] Scanning data3D scans (Found {len(data3d)} scans)...")

        for scan_idx, scan in enumerate(data3d):
            header = e57.get_header(scan_idx)
            point_count = header.point_count
            print(f"[*] Reading Scan {scan_idx} with {point_count} points...")

            data = e57.read_scan(scan_idx, colors=True, ignore_missing_fields=True)

            if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
                import cv2

                r = (data["colorRed"] * 255).astype(np.uint8)
                g = (data["colorGreen"] * 255).astype(np.uint8)
                b = (data["colorBlue"] * 255).astype(np.uint8)

                x = data["cartesianX"]
                y = data["cartesianY"]
                z = data["cartesianZ"]

                # Convert Cartesian to Spherical (equirectangular projection)
                r_dist = np.sqrt(x**2 + y**2 + z**2)
                phi = np.arcsin(np.clip(z / r_dist, -1.0, 1.0))
                theta = np.arctan2(y, x)

                # Map to image resolution 1024x2048
                h, w = 1024, 2048
                u = ((theta + np.pi) / (2 * np.pi) * (w - 1)).astype(np.int32)
                v = ((0.5 - phi / np.pi) * (h - 1)).astype(np.int32)

                pano = np.zeros((h, w, 3), dtype=np.uint8)
                pano[v, u, 0] = b
                pano[v, u, 1] = g
                pano[v, u, 2] = r

                # Inpaint black unmapped pixels
                mask = (
                    (pano[:, :, 0] == 0)
                    & (pano[:, :, 1] == 0)
                    & (pano[:, :, 2] == 0)
                ).astype(np.uint8)
                pano = cv2.inpaint(pano, mask, 3, cv2.INPAINT_TELEA)

                out_path = os.path.join(
                    output_dir, f"projected_scan_{scan_idx}.jpg"
                )
                cv2.imwrite(out_path, pano)
                print(f"[+] Projected and saved: {out_path}")
                image_count += 1

    print(
        f"\n[✓] Done. Total images extracted/generated: {image_count} in '{output_dir}'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 2D panoramas/images from an E57 point cloud file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input .e57 file (e.g. data/scans/cloud_0.e57)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/scans/extracted",
        help="Directory to save extracted images",
    )

    args = parser.parse_args()
    extract_images_from_e57(args.input, args.output)