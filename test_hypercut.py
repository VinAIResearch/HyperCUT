from testing.tester import Tester
from utils import training_utils


def parse_args():
    parser = training_utils.get_basic_parser()

    # Additional parameters
    parser.add_argument("--f_func", type=str, default="ResnetIm2Vec", help="function f")
    parser.add_argument("--g_func", type=str, default="Concat", help="function g")
    parser.add_argument("--out_dim", type=int, default=128, help="Motion vector dimension")

    args = parser.parse_args()
    args.model_name = "HyperCUT"
    args.target_frames = [1, 2, 3, 4, 5, 6, 7]
    num_frames = 2

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

    tester = Tester(args)
    tester.test()
