from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "sis_tween"
    distributed_enabled: bool = True
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    tensorboard_logs_dir: str = "/runs"
    save_progress: bool = True
    save_interval: int = 5
    segmentator_score_threshold : float = 0.1
    epochs: int = 50
    num_seg_classes: int = 6
    autoencoder_path: str = ""
    lr_seg: float = 1e-4
    lr_interp: float = 1e-4
    lr_synth: float = 1e-4


