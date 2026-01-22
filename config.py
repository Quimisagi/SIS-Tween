import argparse
import os
import pickle
import sys
from models.spade.networks.generator import SPADEGenerator
from models.spade.networks.discriminator import MultiscaleDiscriminator

try:
    import utils.utils as utils
except Exception:
    utils = None


def get_parser(train: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training/testing configuration")

    parser.add_argument("--phase", type=str, default="train", help="train, val, test, etc")
    parser.add_argument("--name", type=str, default="sis_tween")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--distributed_enabled", default=True)

    parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    
    # Model configuration
    parser.add_argument("--aspect_ratio", type=float, default=1.0)
    parser.add_argument("--semantic_nc", type=int, default=6)
    parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--isTrain', type=bool, default=train, help='train or test mode')
    
    # Instance-wise features
    parser.add_argument('--no_instance', default=True, action='store_true', help='if specified, do *not* add instance map as input')
    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')

    # Data configuration
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/') 
    parser.add_argument("--dataset_mode", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset.')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file')
    parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size.')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size.')
    parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class.')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    # Logging
    parser.add_argument("--tensorboard_logs_dir", type=str, default="./logs/runs")
    parser.add_argument("--save_progress", action="store_true", default=True)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

    # Architecture details
    parser.add_argument('--class_num', type=int, default=24)
    parser.add_argument("--segmentator_score_threshold", type=float, default=0.1)
    parser.add_argument("--autoencoder_path", type=str, default="")
    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--channels_G", type=int, default=64)
    parser.add_argument("--channels_D", type=int, default=64)
    parser.add_argument("--param_free_norm", type=str, default="syncbatch")
    parser.add_argument("--spade_ks", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--no_spectral_norm", action="store_true")
    parser.add_argument("--no_EMA", action="store_true")
    parser.add_argument("--EMA_decay", type=float, default=0.9999)
    parser.add_argument("--no_3dnoise", action="store_true", default=False)
    parser.add_argument(
        "--active_models",
        type=str,
        nargs="+",
        default=["seg", "interp", "synth"],
        help="Active sub-models",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--lr_interp", type=float, default=1e-4)
    parser.add_argument("--lr_synth", type=float, default=1e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)

    # Losses
    parser.add_argument("--add_vgg_loss", action="store_true")
    parser.add_argument("--no_balancing_inloss", action="store_true", default=False)
    parser.add_argument("--no_labelmix", action="store_true", default=False)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--interp_weight", type=float, default=1.0)
    parser.add_argument("--synth_weight", type=float, default=1.0)

    # Training Control
    if train:
        parser.add_argument("--continue_train", action="store_true")
        parser.add_argument("--which_iter", type=str, default="latest")
        parser.add_argument("--freq_print", type=int, default=1000)
        parser.add_argument("--freq_save_ckpt", type=int, default=20000)
        parser.add_argument("--freq_save_latest", type=int, default=10000)
        parser.add_argument("--freq_smooth_loss", type=int, default=250)
        parser.add_argument("--freq_save_loss", type=int, default=2500)
        parser.add_argument("--freq_fid", type=int, default=5000)
    else:
        parser.add_argument("--results_dir", type=str, default="./results/")
        parser.add_argument("--ckpt_iter", type=str, default="best")

    parser = SPADEGenerator.modify_commandline_options(parser, is_train=True)
    parser = MultiscaleDiscriminator.modify_commandline_options(parser, is_train=True)


    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

    # for training
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

    # the default values for beta1 and beta2 differ by TTUR option
    opt, _ = parser.parse_known_args()
    if opt.no_TTUR:
        parser.set_defaults(beta1=0.5, beta2=0.999)

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

    # for discriminators
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    parser.add_argument('--lambda_kld', type=float, default=0.05)

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
