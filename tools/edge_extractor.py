import cv2
import os
import argparse

from pathlib import Path


def process_image(input_path, output_path, threshold1=100, threshold2=200):
    # Load image in grayscale
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read {input_path}")
        return

    # Apply Canny edge detector
    edges = cv2.Canny(img, threshold1, threshold2)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save edge image as PNG
    cv2.imwrite(str(output_path), edges)


def main():
    parser = argparse.ArgumentParser(description="Extract edges from images in a dataset.")
    parser.add_argument("data_dir", type=str, help="Path to the root data directory")
    parser.add_argument("output_dir", type=str, default="edges", help="Directory to save edge images")
    parser.add_argument("--threshold1", type=int, default=100, help="Canny threshold1")
    parser.add_argument("--threshold2", type=int, default=200, help="Canny threshold2")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for img_path in data_dir.rglob("*.png"):
        # Preserve subfolder structure
        relative_path = img_path.relative_to(data_dir)
        output_path = output_dir / relative_path.with_suffix('.png')

        process_image(img_path, output_path, args.threshold1, args.threshold2)

    print("Edge extraction completed.")


if __name__ == "__main__":
    main()
