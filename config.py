import argparse
import os
import pickle

# Optional utility: safe seed fix
try:
    import utils.utils as utils
except Exception:
    utils = None


def get_parser(train: bool = True) -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Unified training/testing configuration")

    # -------------------- General --------------------
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--name", type=str, default="sis_tween")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--distributed_enabled", default=True)
    parser.add_argument("--aspect_ratio", type=float, default=1.0)
    parser.add_argument("--semantic_nc", type=int, default=6)

    # -------------------- Data --------------------
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_mode", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_flip", action="store_true")

    # -------------------- Saving / Logging --------------------
    parser.add_argument("--tensorboard_logs_dir", type=str, default="./runs")
    parser.add_argument("--save_progress", action="store_true", default=True)
    parser.add_argument("--save_interval", type=int, default=5)

    # -------------------- Model / Architecture --------------------
    parser.add_argument('--class_num', type=int, default=24)
    parser.add_argument("--segmentator_score_threshold", type=float, default=0.1)
    parser.add_argument("--autoencoder_path", type=str, default="")

    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--channels_G", type=int, default=64)
    parser.add_argument("--channels_D", type=int, default=64)


    parser.add_argument(
        "--active_models",
        type=str,
        nargs="+",
        default=["seg", "interp", "synth"],
        help="Active sub-models",
    )

    # -------------------- Optimization --------------------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--lr_interp", type=float, default=1e-4)
    parser.add_argument("--lr_synth", type=float, default=1e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)

    # -------------------- Losses --------------------
    parser.add_argument("--add_vgg_loss", action="store_true")
    parser.add_argument("--no_balancing_inloss", action="store_true", default=False)
    parser.add_argument("--no_labelmix", action="store_true", default=False)

    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--interp_weight", type=float, default=1.0)
    parser.add_argument("--synth_weight", type=float, default=1.0)



    # -------------------- Training Control --------------------
    if train:
        parser.add_argument("--continue_train", action="store_true")
        parser.add_argument("--which_iter", type=str, default="latest")
        parser.add_argument("--freq_print", type=int, default=200)
        parser.add_argument("--freq_save_ckpt", type=int, default=20000)
        parser.add_argument("--freq_save_latest", type=int, default=10000)
        parser.add_argument("--freq_smooth_loss", type=int, default=250)
        parser.add_argument("--freq_save_loss", type=int, default=2500)
        parser.add_argument("--freq_fid", type=int, default=5000)
    else:
        parser.add_argument("--results_dir", type=str, default="./results/")
        parser.add_argument("--ckpt_iter", type=str, default="best")

    return parser

def save_options(opt, parser):
    path = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "opt.txt"), "w") as f:
        for k, v in sorted(vars(opt).items()):
            default = parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            f.write(f"{k:>25}: {str(v):<30}{comment}\n")

    with open(os.path.join(path, "opt.pkl"), "wb") as f:
        pickle.dump(opt, f)


def load_options(opt):
    path = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    return pickle.load(open(path, "rb"))


def load_iter(opt):
    base = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.which_iter == "latest":
        return int(open(os.path.join(base, "latest_iter.txt")).read())
    if opt.which_iter == "best":
        return int(open(os.path.join(base, "best_iter.txt")).read())
    return int(opt.which_iter)


def print_options(opt, parser):
    print("----------------- Options ---------------")
    for k, v in sorted(vars(opt).items()):
        default = parser.get_default(k)
        comment = f"\t[default: {default}]" if v != default else ""
        print(f"{k:>25}: {str(v):<30}{comment}")
    print("----------------- End -------------------")


def read_arguments(train: bool = True):
    parser = get_parser(train)
    opt = parser.parse_args()

    if train:
        if getattr(opt, "continue_train", False):
            prev = load_options(opt)
            for k, v in vars(prev).items():
                if hasattr(opt, k):
                    parser.set_defaults(**{k: v})
        opt = parser.parse_args()
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)

    opt.phase = "train" if train else "test"

    if utils is not None:
        utils.fix_seed(opt.seed)

    print_options(opt, parser)
    if train:
        save_options(opt, parser)

    return opt
