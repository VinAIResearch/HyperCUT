from training.trainer import Trainer
from utils import training_utils


def parse_args():
    parser = training_utils.get_basic_parser()

    # Additional parameters
    parser.add_argument("--f_func", type=str, default="ResnetIm2Vec", help="function f")
    parser.add_argument("--g_func", type=str, default="Concat", help="function g")
    parser.add_argument("--out_dim", type=int, default=128, help="Motion vector dimension")

    parser.add_argument("--val_epoch", type=int, default=5, help="Frequecy of validation and saving epoch")
    parser.add_argument("--train_step", type=int, default=100, help="Frequecy of logging learning_rate and loss")

    args = parser.parse_args()
    args.model_name = "HyperCUT"
    args.display_step = 50
    num_frames = 2
    args.target_frames = [1, 2, 3, 4, 5, 6, 7]
    args.exp_name = f"HyperCUT_{args.f_func}_{args.g_func}_{num_frames}frames_dim{args.out_dim}_{args.dataset_name}"

    args.use_flow = args.g_func == "flow"

    args.model_kwargs = {
        "f_func": args.f_func,
        "g_func": args.g_func,
        "num_frames": num_frames,
        "out_dim": args.out_dim,
    }

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.training_loop()
