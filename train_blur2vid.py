from training.trainer import Trainer
from utils import training_utils


def parse_args():
    parser = training_utils.get_basic_parser()

    # Additional parameters
    parser.add_argument("--backbone", type=str, required=True, help="backbone")
    parser.add_argument("--loss_type", type=str, required=True, help="Loss type")
    parser.add_argument("--target_frames", nargs="+", type=int, default=None, help="Target frames")

    parser.add_argument("--val_epoch", type=int, default=5, help="Frequecy of validation and saving epoch")
    parser.add_argument("--train_step", type=int, default=100, help="Frequecy of logging learning_rate and loss")
    parser.add_argument("--display_step", type=int, default=100, help="Frequecy of image visualization")

    args = parser.parse_args()
    args.model_name = "Blur2Vid"
    num_frames = 2
    args.exp_name = f"Blur2Vid_{args.backbone}_{args.loss_type}_{num_frames}frames_{args.dataset_name}"

    args.model_kwargs = {
        "backbone_name": args.backbone,
        "backbone_kwargs": {
            "num_frames": num_frames,
            "stage1_path": "checkpoints/purohit.pth",
            "loss_kwargs": {
                "loss_type": args.loss_type,
                "mu": 0.02,
                "HyperCUT": {
                    "pretrained_path": args.hypercut_path,
                    "f_func": "ResnetIm2Vec",
                    "g_func": "Concat",
                    "num_frames": num_frames,
                    "out_dim": 128,
                    "alpha": 0.2,
                },
            },
        },
    }

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.training_loop()
