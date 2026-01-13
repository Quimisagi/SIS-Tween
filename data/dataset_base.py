import os
from PIL import Image


class DatasetBase:
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.mode = mode

        train_img_root = os.path.join(root_dir, "train", "images")
        if not os.path.isdir(train_img_root):
            raise FileNotFoundError(f"Missing train/images: {train_img_root}")

        self.classes = sorted([
            d for d in os.listdir(train_img_root)
            if os.path.isdir(os.path.join(train_img_root, d))
        ])

        if len(self.classes) == 0:
            raise RuntimeError("No class folders found under train/images")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, lbl_path, edg_path, basename, class_id = self.samples[index]

        image = self._load_image(img_path)
        label = self._load_image(lbl_path)
        edge = self._load_image(edg_path)

        return {
            "basename": basename,
            "image": image,
            "label": label,
            "edge": edge,
            "class": class_id,
        }

    def _build_index(self):
        samples = []

        for split in ["train", "val", "test"]:
            if self.mode not in [split, "all"]:
                continue

            img_root = os.path.join(self.root_dir, split, "images")
            lbl_root = os.path.join(self.root_dir, split, "labels")
            edge_root = os.path.join(self.root_dir, split, "edges")

            if not os.path.isdir(img_root):
                continue

            for cls in sorted(os.listdir(img_root)):
                img_dir = os.path.join(img_root, cls)
                lbl_dir = os.path.join(lbl_root, cls)
                edge_dir = os.path.join(edge_root, cls)

                if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir) and os.path.isdir(edge_dir)):
                    continue

                class_id = self.class_to_idx[cls]

                img_files = sorted(os.listdir(img_dir))
                lbl_files = sorted(os.listdir(lbl_dir))
                edge_files = sorted(os.listdir(edge_dir))

                for img_f, lbl_f, edg_f in zip(img_files, lbl_files, edge_files):
                    samples.append(
                        (
                            os.path.join(img_dir, img_f),
                            os.path.join(lbl_dir, lbl_f),
                            os.path.join(edge_dir, edg_f),
                            f"{split}/{cls}/{img_f}",
                            class_id,
                        )
                    )

        return samples

    def _load_image(self, path):
        with Image.open(path) as img:
            return img.copy()
