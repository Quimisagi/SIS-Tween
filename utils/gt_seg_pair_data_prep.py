import random
from pathlib import Path
import shutil

# ---- CONFIG ----
SOURCE_DIR = Path("/home/quimisagi/Daigaku/SIS_Dataset")
DEST_DIR = Path("/media/quimisagi/KIOXIA/Own/November/sis_data")

GT_SRC = SOURCE_DIR / "GT"
SEG_SRC = SOURCE_DIR / "Seg"

GT_DEST = DEST_DIR / "GT"
SEG_DEST = DEST_DIR / "Seg"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_unique_filename(rel_path: Path):
    """
    Convert relative path into a unique filename by replacing 
    directory separators with underscores.
    Example:
        humans/set1/img001.png → humans_set1_img001.png
    """
    return "_".join(rel_path.parts)


def paired_files(root_gt: Path, root_seg: Path):
    """
    Finds all GT images and returns (relative_path, GT file, Seg file)
    Only yields pairs where both files exist.
    """
    for gt_file in root_gt.rglob("*"):
        if gt_file.is_file():
            rel = gt_file.relative_to(root_gt)
            seg_file = root_seg / rel

            if seg_file.exists() and seg_file.is_file():
                yield rel, gt_file, seg_file
            else:
                print(f"[WARN] Missing Seg file for: {rel}")


def main():
    print("Copying all dataset images into single folders...")

    ensure_dir(GT_DEST)
    ensure_dir(SEG_DEST)

    for rel_path, gt_img, seg_img in paired_files(GT_SRC, SEG_SRC):

        # create unique flattened name
        unique_name = make_unique_filename(rel_path)

        # final output paths
        out_gt = GT_DEST / unique_name
        out_seg = SEG_DEST / unique_name

        shutil.copy2(gt_img, out_gt)
        shutil.copy2(seg_img, out_seg)

        print(f"[OK] {rel_path} → {unique_name}")

    print("Done!")


if __name__ == "__main__":
    main()
