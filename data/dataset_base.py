import os
from PIL import Image


class DatasetBase:
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, lbl_path, basename = self.samples[index]

        image = self._load_image(img_path)
        label = self._load_image(lbl_path)

        return {
            "basename": basename,
            "image": image,
            "label": label,
        }

    def _build_index(self):
        samples = []

        for split in ["train", "val", "test"]:
            if self.mode not in [split, "all"]:
                continue

            img_root = os.path.join(self.root_dir, split, "images")
            lbl_root = os.path.join(self.root_dir, split, "labels")

            if not os.path.isdir(img_root):
                continue

            for seq in sorted(os.listdir(img_root)):
                img_dir = os.path.join(img_root, seq)
                lbl_dir = os.path.join(lbl_root, seq)

                if not os.path.isdir(img_dir):
                    continue

                img_files = sorted(os.listdir(img_dir))
                lbl_files = sorted(os.listdir(lbl_dir))

                for img_f, lbl_f in zip(img_files, lbl_files):
                    samples.append(
                        (
                            os.path.join(img_dir, img_f),
                            os.path.join(lbl_dir, lbl_f),
                            f"{split}/{seq}",
                        )
                    )

        return samples

    def _load_image(self, path):
        with Image.open(path) as img:
            return img.copy()
